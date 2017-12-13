/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.recommendation

import java.{util => ju}
import breeze.linalg.{DenseVector => BrzVector}
import breeze.optimize.{FirstOrderMinimizer, DiffFunction}
import breeze.optimize.proximal.Constraint._
import breeze.optimize.proximal.NonlinearMinimizer
import com.github.fommil.netlib.BLAS.{getInstance=>blas}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.recommendation.ALS._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.optimization.LossType._
import org.apache.spark.mllib.optimization.{Gradient, LeastSquaresGradient, LogLikelihoodGradient}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.random.XORShiftRandom
import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.hashing._
import ALS.Rating
import breeze.stats.distributions.Rand

object ALM extends Logging {
  // TO DO: Generalize it to Vector so that we can shuffle both Dense (low ranks) and Sparse (large ranks)
  // mllib vectors
  private type FactorBlock = Array[Vector]

  // Possible Gradients are LeastSquareGradient and LogLikelihoodGradient
  private [recommendation] class ALMCost(rank: Int,
                                         gradient: Gradient) extends DiffFunction[BrzVector[Double]] with Serializable {
    val countBuilder = mutable.ArrayBuilder.make[Double]
    val factorBuilder = mutable.ArrayBuilder.make[Vector]

    val cumGradient = new Array[Double](rank)

    // For mllib Gradient re-use
    val cumGradientV = Vectors.dense(cumGradient)
    // For Breeze Optimizer re-use
    val cumGradientBrz = BrzVector(cumGradient)

    var n = 0

    def reset(): Unit = {
      countBuilder.clear()
      factorBuilder.clear()
      n = 0
    }

    def add(a: Vector, b: Float): ALMCost = {
      require(a.size == rank, s"ALMCost:add normal equation and rank mismatch")
      countBuilder += b
      factorBuilder += a
      n += 1
      this
    }

    override def calculate(brzWeights: BrzVector[Double]): (Double, BrzVector[Double]) = {
      require(n > 0, s"ALMCost:calculate sample should be greater than zero")
      var i = 0
      val weights = Vectors.fromBreeze(brzWeights)
      var objective = 0.0

      val counts = countBuilder.result()
      val factors = factorBuilder.result()

      // workspace shared with mllib Gradient and Breeze Optimizer interface
      // TO DO: Do we need to some normalization based on number of normal equations ?
      ju.Arrays.fill(cumGradient, 0.0)
      while (i < n) {
        objective += gradient.compute(factors(i), counts(i), weights, cumGradientV)
        i += 1
      }
      (objective, cumGradientBrz)
    }
  }

  private [recommendation] class ALMSolver(rank: Int,
                                           loss: LossType = LEASTSQUARE,
                                           constraint: Constraint = PROBABILITYSIMPLEX) extends Serializable {
    private val gradient = loss match {
      case LEASTSQUARE => new LeastSquaresGradient()
      case LOGLIKELIHOOD => new LogLikelihoodGradient()
    }

    private val init = BrzVector.rand[Double](rank, Rand.gaussian(0,1))
    val cost = new ALMCost(rank, gradient)

    private var solver: FirstOrderMinimizer[BrzVector[Double], DiffFunction[BrzVector[Double]]] = _
    private var initialized: Boolean = false

    // ADMM based proximal solver
    //private val proximal = ProjectProbabilitySimplex(1.0)
    //private val solver = new NonlinearMinimizer(proximal)

    // SPG based Projected gradient solver
    def initialize = {
      if (!initialized) {
        solver = NonlinearMinimizer(rank, constraint, 1.0)
        initialized = true
      } else {
        require(this.rank == rank)
      }
    }

    // lambda is for L2 regularization which is added for robustness (similar to Tikhonov regularization)
    // TO DO : Fold L2 within ALMCost and clean DiffFunction.withL2Regularization
    def solve(lambda: Double): Vector = {
      val regularizedCost = DiffFunction.withL2Regularization(cost, lambda)
      initialize
      val x = solver.minimize(regularizedCost, init)
      //TO DO : Convert dense to sparse based on the constraint's effect if applicable
      Vectors.fromBreeze(x)
    }
  }

  /**
   * Initializes factors randomly given the in-link blocks.
   *
   * @param inBlocks in-link blocks
   * @param rank rank
   * @return initialized factor blocks
   */
  private [recommendation] def initialize[ID](
    inBlocks: RDD[(Int, InBlock[ID])],
    rank: Int,
    seed: Long): RDD[(Int, FactorBlock)] = {
    // Choose a unit vector uniformly at random from the unit sphere, but from the
    // "first quadrant" where all elements are nonnegative. This can be done by choosing
    // elements distributed as Normal(0,1) and taking the absolute value, and then normalizing.
    // This appears to create factorizations that have a slightly better reconstruction
    // (<1%) compared picking elements uniformly at random in [0,1].

    // TO DO : This is tricky. When we initialize for sparse ranks make sure that it is not all zeros
    inBlocks.map { case (srcBlockId, inBlock) =>
      val random = new XORShiftRandom(byteswap64(seed ^ srcBlockId))
      val factors = Array.fill(inBlock.srcIds.length) {
        val factor = Array.fill(rank)(random.nextGaussian().toFloat)
        val nrm = blas.snrm2(rank, factor, 1)
        blas.sscal(rank, 1.0f / nrm, factor, 1)
        // Basically generate a good sparse vector for initialization, is anything better than zero ?
        Vectors.dense(factor.map{_.toDouble})
      }
      (srcBlockId, factors)
    }
  }
  /**
   * Compute dst factors by constructing and solving constrained minimization problems.
   *
   * @param srcFactorBlocks src factors
   * @param srcOutBlocks src out-blocks
   * @param dstInBlocks dst in-blocks
   * @param rank rank
   * @param regParam regularization constant
   * @param srcEncoder encoder for src local indices
   * @param solver solver for nonlinear minimization problems
   *
   * @return dst factors
   */
  private def computeFactors[ID](
    srcFactorBlocks: RDD[(Int, FactorBlock)],
    srcOutBlocks: RDD[(Int, OutBlock)],
    dstInBlocks: RDD[(Int, InBlock[ID])],
    rank: Int,
    regParam: Double,
    srcEncoder: LocalIndexEncoder,
    solver: ALMSolver): RDD[(Int, FactorBlock)] = {
    val numSrcBlocks = srcFactorBlocks.partitions.length
    val srcOut = srcOutBlocks.join(srcFactorBlocks).flatMap {
      case (srcBlockId, (srcOutBlock, srcFactors)) =>
        srcOutBlock.view.zipWithIndex.map { case (activeIndices, dstBlockId) =>
          (dstBlockId, (srcBlockId, activeIndices.map(idx => srcFactors(idx))))
        }
    }
    val merged = srcOut.groupByKey(new ALSPartitioner(dstInBlocks.partitions.length))
    dstInBlocks.join(merged).mapValues {
      case (InBlock(dstIds, srcPtrs, srcEncodedIndices, ratings), srcFactors) =>
        val sortedSrcFactors = new Array[FactorBlock](numSrcBlocks)
        srcFactors.foreach { case (srcBlockId, factors) =>
          sortedSrcFactors(srcBlockId) = factors
        }
        val dstFactors = new Array[Vector](dstIds.length)
        var j = 0
        while (j < dstIds.length) {
          solver.cost.reset()
          var i = srcPtrs(j)
          while (i < srcPtrs(j + 1)) {
            val encoded = srcEncodedIndices(i)
            val blockId = srcEncoder.blockId(encoded)
            val localIndex = srcEncoder.localIndex(encoded)
            val srcFactor = sortedSrcFactors(blockId)(localIndex)
            val rating = ratings(i)
            solver.cost.add(srcFactor, rating)
            i += 1
          }
          dstFactors(j) = solver.solve(regParam)
          j += 1
        }
        dstFactors
    }
  }

  /**
   * Implementation of Alternating Minimization algorithm for cases where ranks
   * are large (generating gram matrix is difficult) and we want to support
   * convex loss functions like least square (validation purpose), loglikelihood and
   * logistic loss
   */
  def train[ID: ClassTag]( // scalastyle:ignore
    ratings: RDD[Rating[ID]],
    numUserBlocks: Int = 10,
    numItemBlocks: Int = 10,
    rank: Int = 10,
    maxIter: Int = 10,
    loss: LossType = LEASTSQUARE,
    userConstraint: Constraint = SMOOTH,
    itemConstraint: Constraint = SMOOTH,
    userRegParam: Double = 1.0,
    itemRegParam: Double = 1.0,
    intermediateRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    finalRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    seed: Long = 0L)(
    implicit ord: Ordering[ID]): (RDD[(ID, Vector)], RDD[(ID, Vector)]) = {
    require(intermediateRDDStorageLevel != StorageLevel.NONE,
      "ALS is not designed to run without persisting intermediate RDDs.")
    val userPart = new ALSPartitioner(numUserBlocks)
    val itemPart = new ALSPartitioner(numItemBlocks)
    val userLocalIndexEncoder = new LocalIndexEncoder(userPart.numPartitions)
    val itemLocalIndexEncoder = new LocalIndexEncoder(itemPart.numPartitions)

    val userSolver = new ALMSolver(rank, loss, userConstraint)
    val itemSolver = new ALMSolver(rank, loss, itemConstraint)

    val blockRatings = ALS.partitionRatings(ratings, userPart, itemPart)
      .persist(intermediateRDDStorageLevel)
    val (userInBlocks, userOutBlocks) =
      ALS.makeBlocks("user", blockRatings, userPart, itemPart, intermediateRDDStorageLevel)
    // materialize blockRatings and user blocks
    userOutBlocks.count()
    val swappedBlockRatings = blockRatings.map {
      case ((userBlockId, itemBlockId), ALS.RatingBlock(userIds, itemIds, localRatings)) =>
        ((itemBlockId, userBlockId), ALS.RatingBlock(itemIds, userIds, localRatings))
    }
    val (itemInBlocks, itemOutBlocks) =
      ALS.makeBlocks("item", swappedBlockRatings, itemPart, userPart, intermediateRDDStorageLevel)
    // materialize item blocks
    itemOutBlocks.count()
    val seedGen = new XORShiftRandom(seed)

    var userFactors = ALM.initialize(userInBlocks, rank, seedGen.nextLong())
    var itemFactors = ALM.initialize(itemInBlocks, rank, seedGen.nextLong())

    for (iter <- 0 until maxIter) {
      itemFactors = computeFactors(userFactors, userOutBlocks, itemInBlocks, rank, itemRegParam,
        userLocalIndexEncoder, solver = itemSolver)
      userFactors = computeFactors(itemFactors, itemOutBlocks, userInBlocks, rank, userRegParam,
        itemLocalIndexEncoder, solver = userSolver)
    }

    val userIdAndFactors = userInBlocks
      .mapValues(_.srcIds)
      .join(userFactors)
      .mapPartitions({ items =>
      items.flatMap { case (_, (ids, factors)) =>
        ids.view.zip(factors)
      }
      // Preserve the partitioning because IDs are consistent with the partitioners in userInBlocks
      // and userFactors.
    }, preservesPartitioning = true)
      .setName("userFactors")
      .persist(finalRDDStorageLevel)
    val itemIdAndFactors = itemInBlocks
      .mapValues(_.srcIds)
      .join(itemFactors)
      .mapPartitions({ items =>
      items.flatMap { case (_, (ids, factors)) =>
        ids.view.zip(factors)
      }
    }, preservesPartitioning = true)
      .setName("itemFactors")
      .persist(finalRDDStorageLevel)
    if (finalRDDStorageLevel != StorageLevel.NONE) {
      userIdAndFactors.count()
      itemFactors.unpersist()
      itemIdAndFactors.count()
      userInBlocks.unpersist()
      userOutBlocks.unpersist()
      itemInBlocks.unpersist()
      itemOutBlocks.unpersist()
      blockRatings.unpersist()
    }
    (userIdAndFactors, itemIdAndFactors)
  }
}
