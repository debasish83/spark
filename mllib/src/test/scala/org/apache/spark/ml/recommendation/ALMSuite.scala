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

import org.apache.spark.ml.recommendation.ALM.{ALMSolver, ALMCost}
import org.apache.spark.mllib.linalg.{Vectors}
import org.apache.spark.mllib.optimization.{LogLikelihoodGradient, LeastSquaresGradient}
import breeze.linalg.{DenseVector => BrzVector, DenseMatrix => BrzMatrix}
import breeze.optimize.proximal.{ProjectBox, ProjectProbabilitySimplex, NonlinearMinimizer, QuadraticMinimizer}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.util.TestingUtils._
import breeze.optimize.proximal.Constraint._
import org.apache.spark.internal.Logging
import org.scalatest.FunSuite
import spire.syntax.cfor._

class ALMSuite extends FunSuite with MLlibTestSparkContext with Logging {

  // For deterministic results over tests we generate a sample data as follows:
  // val rand = Rand.gaussian(0, 1)
  // val factors = BrzMatrix.rand[Double](5, 5, rand)
  // val ratings = BrzVector.rand[Double](5, rand)

  val factors = new BrzMatrix(5, 5,
    Array(1.9169634885999849, -0.790646554794345, -0.37021287423830357, -0.28238027389691134, -0.4488394652402634,
      0.055814133013590644, 1.493717265858178, -1.0168911621377017, -1.3574976260348481, 1.2395574515777905,
      -1.3146090263022396, -0.7762636319158648, -1.0381371111576219, -0.1014153505826655, -0.505444400208315,
      0.5754760706002274, -0.5096433859266171, -0.23426234976768193, -1.1703494868024424, 0.21561735900991533,
      0.9764428943466384, 0.06509377709123572, -1.044756921029536, 0.0028390638836486384, -2.2685155886081425))

  val ratings =
    BrzVector(Array(-0.6168289352061995, -1.7179326630111105, 1.2549766272269893, 0.14124759379460353, -0.06855961426721971))

  val gram = factors.t * factors
  val q = factors.t * ratings
  q *= -1.0

  val init = BrzVector.zeros[Double](5)
  val cost = QuadraticMinimizer.Cost(gram, q)

  test("ALMCost validation for least squares loss") {
    /*
    octave:18> A
    A =

      -1.77322   1.25092   0.77880
    1.79053  -1.04103   0.35574
    0.82641  -1.19828  -0.11098
    0.84072  -2.06089  -0.49674

    octave:19> x
    x = -0.65095 0.64076 0.81478

    octave:20> b
    b = 0.48916 0.62934 0.58653 0.77959

    octave:21> grad = A'*(A*x - b)
    grad = -11.8195   13.5556    2.5999

    octave:23> 0.5*(A*x - b)'*(A*x - b)
      ans =  11.190
    */
    val gradient = new LeastSquaresGradient
    val almCost = new ALMCost(3, gradient)
    almCost.add(Vectors.dense(-1.77322, 1.25092, 0.77880), 0.48916f)
      .add(Vectors.dense(1.79053, -1.04103, 0.35574), 0.62934f)
      .add(Vectors.dense(0.82641, -1.19828, -0.11098), 0.58653f)
      .add(Vectors.dense(0.84072, -2.06089, -0.49674), 0.77959f)
    assert(almCost.n === 4)
    val weights = BrzVector(-0.65095, 0.64076, 0.81478)
    val (obj, grad) = almCost.calculate(weights)

    assert(obj ~= 11.190 relTol 1e-4)
    assert(Vectors.dense(grad.data) ~== Vectors.dense(-11.8195, 13.5556, 2.5999) relTol 1e-4)
  }

  test("ALMCost validation for loglikelihood loss") {
    val gradient = new LogLikelihoodGradient
    val almCost = new ALMCost(2, gradient)
      .add(Vectors.dense(1.0, 2.0), 4.0f)
      .add(Vectors.dense(1.0, 3.0), 9.0f)
      .add(Vectors.dense(1.0, 4.0), 16.0f)
    assert(almCost.n === 3)
    val weights = BrzVector(0.3, 0.6)
    //obj = 4.0*log(1.5) + 9.0*log(2.1) + 16.0*log(2.7)
    //grad (4.0/1.5)*(1.0, 2.0) (9.0/2.1)*(1.0, 3.0) (16.0/2.7)*(1.0, 4.0)
    val grad1 = -1.0*(4.0 / 1.5 + 9.0 / 2.1 + 16.0 / 2.7)
    val grad2 = -1.0*(8.0 / 1.5 + 27.0 / 2.1 + 64.0 / 2.7)
    val golden = Vectors.dense(grad1, grad2)
    val (obj, grad) = almCost.calculate(weights)
    assert(obj ~= -24.191 relTol 1e-4)
    assert(Vectors.dense(grad.data) ~== golden relTol 1e-4)
  }

  test("ALMCost gradient validation for loglikehood loss") {
    val gradient = new LogLikelihoodGradient
    val almCost = new ALMCost(2, gradient)
      .add(Vectors.dense(1.0, 2.0), 4.0f)
      .add(Vectors.dense(1.0, 3.0), 9.0f)
      .add(Vectors.dense(1.0, 4.0), 16.0f)
    assert(almCost.n === 3)
    val weights = BrzVector(0.3, 0.6)
    val (obj, grad) = almCost.calculate(weights)
    val eps = 1e-8
    weights(0) += eps
    val numerical1 = (almCost.valueAt(weights) - obj) / eps
    weights(0) -= eps
    weights(1) += eps
    val numerical2 = (almCost.valueAt(weights) - obj) / eps

    assert(numerical1 ~= grad(0) relTol 1e-4)
    assert(numerical2 ~= grad(1) relTol 1e-4)
  }

  test("ALMCost consistency with multiple invocations") {
    val gradient = new LogLikelihoodGradient
    val almCost = new ALMCost(2, gradient)
      .add(Vectors.dense(3.0, 2.0), 2.0f)
      .add(Vectors.dense(4.0, 5.0), 1.0f)
    almCost.reset()
    almCost.add(Vectors.dense(1.0, 2.0), 4.0f)
      .add(Vectors.dense(1.0, 3.0), 9.0f)
      .add(Vectors.dense(1.0, 4.0), 16.0f)

    val weights = BrzVector(0.3, 0.6)
    //obj = 4.0*log(1.5) + 9.0*log(2.1) + 16.0*log(2.7)
    //grad (4.0/1.5)*(1.0, 2.0) (9.0/2.1)*(1.0, 3.0) (16.0/2.7)*(1.0, 4.0)
    val grad1 = -1.0*(4.0 / 1.5 + 9.0 / 2.1 + 16.0 / 2.7)
    val grad2 = -1.0*(8.0 / 1.5 + 27.0 / 2.1 + 64.0 / 2.7)
    val golden = Vectors.dense(grad1, grad2)
    val (obj, grad) = almCost.calculate(weights)
    assert(obj ~= -24.191 relTol 1e-4)
    assert(Vectors.dense(grad.data) ~== golden relTol 1e-4)
  }

  test("NonlinearMinimizer compared to Quadratic Minimization with Equality") {
    init := 0.0
    val nl = new NonlinearMinimizer(proximal = ProjectProbabilitySimplex(1.0))

    val x = nl.minimize(cost, init)

    val quadraticx = QuadraticMinimizer(5, EQUALITY).minimize(gram, q)

    val qpObj = QuadraticMinimizer.computeObjective(gram, q, quadraticx)
    val nlObj = cost.valueAt(BrzVector(x.toArray))

    assert(nlObj ~== qpObj absTol 1e-3)
    assert(Vectors.dense(quadraticx.data) ~== Vectors.dense(x.data) absTol 1e-3)
  }

  test("NonlinearMinimizer compared to Quadratic Minimization with bounds") {
    init := 0.0
    val lb = BrzVector.zeros[Double](5)
    val ub = BrzVector.ones[Double](5)

    val proximal = ProjectBox(lb, ub)

    val nl = new NonlinearMinimizer(proximal)
    val x = nl.minimize(cost, init)
    val brzx = BrzVector(x.toArray)

    val quadraticx = QuadraticMinimizer(5, BOX).minimize(gram, q)

    val qpBoxObj = QuadraticMinimizer.computeObjective(gram, q, quadraticx)
    val nlBoxObj = cost.valueAt(brzx)

    assert(qpBoxObj ~== nlBoxObj absTol 1e-3)
    assert(Vectors.dense(quadraticx.data) ~== Vectors.dense(x.data) absTol 1e-3)
  }

  test("ALMSolver gradient validation compared to Quadratic Model") {
    val almSolver = new ALMSolver(3)

    almSolver.cost.add(Vectors.dense(-1.77322, 1.25092, 0.77880), 0.48916f)
      .add(Vectors.dense(1.79053, -1.04103, 0.35574), 0.62934f)
      .add(Vectors.dense(0.82641, -1.19828, -0.11098), 0.58653f)
      .add(Vectors.dense(0.84072, -2.06089, -0.49674), 0.77959f)

    val quadFactors = new BrzMatrix[Double](4, 3,
      Array(-1.77322, 1.79053, 0.82641, 0.84072,
        1.25092, -1.04103, -1.19828, -2.06089,
        0.77880, 0.35574, -0.11098, -0.49674))

    val quadRatings = BrzVector(0.48916, 0.62934, 0.58653, 0.77959)

    //0.5*||Ax - b||^2 = 0.5*(Ax - b)'(Ax - b) = 0.5*x'(A'A)*x - (A'b)*x
    val gramQuad = (quadFactors.t * quadFactors)
    val qQuad = quadFactors.t * quadRatings
    qQuad *= -1.0

    val weights = BrzVector(-0.65095, 0.64076, 0.81478)

    val (almTestObj, almTestGrad) = almSolver.cost.calculate(weights)
    val testDiff = quadFactors * weights - quadRatings
    val qpTestObj = 0.5 * (testDiff.t * testDiff)
    val qpTestGrad: BrzVector[Double] = QuadraticMinimizer.Cost(gramQuad, qQuad).gradientAt(weights)

    assert(almTestObj ~= qpTestObj absTol 1e-4)
    assert(Vectors.dense(almTestGrad.data) ~== Vectors.dense(qpTestGrad.data) absTol 1e-4)
  }

  test("ALMSolver compared to Quadratic Minimization with Equality") {
    val almSolver = new ALMSolver(3)

    almSolver.cost.add(Vectors.dense(-1.77322, 1.25092, 0.77880), 0.48916f)
      .add(Vectors.dense(1.79053, -1.04103, 0.35574), 0.62934f)
      .add(Vectors.dense(0.82641, -1.19828, -0.11098), 0.58653f)
      .add(Vectors.dense(0.84072, -2.06089, -0.49674), 0.77959f)

    val quadFactors = new BrzMatrix[Double](4, 3,
      Array(-1.77322, 1.79053, 0.82641, 0.84072,
        1.25092, -1.04103, -1.19828, -2.06089,
        0.77880, 0.35574, -0.11098, -0.49674))

    val quadRatings = BrzVector(0.48916, 0.62934, 0.58653, 0.77959)

    val x = almSolver.solve(0.0)
    val brzx = BrzVector(x.toArray)
    val almObj = almSolver.cost.valueAt(brzx)

    //0.5*||Ax - b||^2 = 0.5*(Ax - b)'(Ax - b) = 0.5*x'(A'A)*x - (A'b)*x
    val gramQuad = (quadFactors.t * quadFactors)
    val qQuad = quadFactors.t * quadRatings
    qQuad *= -1.0

    val quadraticx = QuadraticMinimizer(3, EQUALITY).minimize(gramQuad, qQuad)

    val diff = quadFactors * quadraticx - quadRatings
    val qpObj = 0.5 * (diff.t * diff)

    assert(almObj ~== qpObj absTol 1e-4)
    assert(Vectors.dense(quadraticx.data) ~== x absTol 1e-4)
  }

  // TO DO : Add similar tests for asymmetric matrix
  test("ALMSolver compared to Quadratic Minimization with Equality on symmetric gaussian data") {
    val almSolver = new ALMSolver(5)
    cforRange(0 until factors.rows) { i =>
      val row = factors(i, ::).t
      almSolver.cost.add(Vectors.dense(row.toArray), ratings(i).toFloat)
    }

    val x = almSolver.solve(0.0)
    val brzx = BrzVector(x.toArray)
    val almObj = almSolver.cost.valueAt(brzx)

    val quadraticx = QuadraticMinimizer(5, EQUALITY).minimize(gram, q)

    val diff = factors * quadraticx - ratings
    val qpObj = 0.5 * (diff.t * diff)

    assert(almObj ~== qpObj absTol 1e-3)
    assert(Vectors.dense(quadraticx.data) ~== x absTol 1e-3)
  }

  test("ALMSolver compared to Quadratic Minimization with Bounds symmetric gaussian data") {
    val almSolver = new ALMSolver(5, constraint = BOX)
    cforRange(0 until factors.rows) { i =>
      val row = factors(i, ::).t
      almSolver.cost.add(Vectors.dense(row.toArray), ratings(i).toFloat)
    }
    val x = almSolver.solve(0.0)
    val brzx = BrzVector(x.toArray)
    val almObj = almSolver.cost.valueAt(brzx)

    val quadraticx = QuadraticMinimizer(5, BOX).minimize(gram, q)

    val diff = factors * quadraticx - ratings
    val qpObj = 0.5 * (diff.t * diff)

    assert(almObj ~== qpObj absTol 1e-3)
    assert(Vectors.dense(quadraticx.data) ~== x absTol 1e-3)
  }
}
