/////////////////////////////////////////////////////////////////
// This class implements the KD-Fix confidence estimation method.
// KD-Fix stands for K-Draws by Fixed Standard deviation.
// Given the weights vector of the trained model, K additional
// weight vectors are drawn from it using Gaussian distribution
// per parameter using a fixed, user defined Std-dev.
// Each of the K weight vectors is used to generate an alternative
// parse for an input instance. Finally, the confidence score of
// each edge in the parse tree is defined to be the fraction of
// alternatives that agree with the base model parse regarding that
// edge.
//
// Example using K=3
// The trained model M --> 3 samples --> M1, M2, M3
// Parse the instance using the trained model M and the 3
// additional models:
//
// instance words:   1,   2,   3,..., 18,  19
// heads by M    :   2,   4,   0,..., 14,  18
//                ----------------------------
// heads by M1   :   2,   5,   0,..., 14,  16
// heads by M2   :   2,   4,   0,..., 14,  18
// heads by M3   :   2,   5,   1,..., 14,  18
//                ----------------------------
// Agree-count   :   3,   1,   2,...,  3,   2
// Confidence    :   1, 1/3, 2/3,...,  1, 2/3
//
// Author: Avihai Mejer, 2011
/////////////////////////////////////////////////////////////////

package mstparser;

import java.util.Random;

public class ConfidenceEstimatorKDFix extends ConfidenceEstimator {
  // The number of parses to use for estimation.
  int k;

  // The Std-Dev to use for drawing the K parameter vectors.
  double stddev;

  Parameters parameters[];

  DependencyParser depParser;

  public ConfidenceEstimatorKDFix(double stddev, int k, DependencyParser depParser) {
    this.stddev = stddev;
    this.k = k;
    this.depParser = depParser;
    drawKParameterVectors();
  }

  // If the model parameters of the dependency parser are updated then
  // this method must be called to re-draw the K parameter vectors.
  public void drawKParameterVectors() {
    Random rand = new Random();
    // This is the base model parameters.
    double[] modelWeights = depParser.getParams().parameters;
    int numParams = modelWeights.length;

    // Draw K parameter vectors
    parameters = new Parameters[k];
    for (int i = 0; i < k; i++) {
      double[] params = new double[numParams];

      // Draw each of the parameters according to the base model
      // parameter and the required Std-Dev.
      for (int j = 0; j < numParams; j++) {
        double gauss = rand.nextGaussian();
        double mean = modelWeights[j];
        params[j] = mean + gauss * stddev;
      }
      parameters[i] = new Parameters(params);
    }
  }

  // /////////////////////////////////////////////////////////////////
  // Compute the confidence score for each predicted edge.
  // /////////////////////////////////////////////////////////////////
  @Override
  public double[] estimateConfidence(DependencyInstance inst) {
    // Parse the instance using the base parser parameters.
    int[] predictedHeads = new int[inst.heads.length - 1];
    depParser.decode(inst, 1, depParser.getParams(), predictedHeads);

    int[][] alternativeHeads = new int[k][predictedHeads.length];
    // Generate K alternative parses using the K sampled parameters vectors.
    // Note: Using multi-threaded implementation.
    produceKAlternatives_MT(inst, alternativeHeads);

    return confidenceScoresByAgreement(predictedHeads, alternativeHeads);
  }

  double[] confidenceScoresByAgreement(int[] prediction, int[][] alternatives) {
    // For each prediction count the number of alternatives
    // that agree with the prediction.
    double[] confidenceScores = new double[prediction.length];
    for (int h = 0; h < prediction.length; h++) {
      int agreeCount = 0;
      for (int i = 0; i < k; i++) {
        if (prediction[h] == alternatives[i][h]) {
          agreeCount++;
        }
      }

      // The confidence score is the fraction of alternatives that
      // agree with the prediction.
      confidenceScores[h] = (double) agreeCount / k;
    }
    return confidenceScores;
  }

  // /////////////////////////////////////////////////////////////////
  // Multi-threaded implementation of produce-K-Alternatives.
  // /////////////////////////////////////////////////////////////////
  void produceKAlternatives_MT(DependencyInstance inst, int[][] alternativeHeads) {
    DecoderRunner[] runners = new DecoderRunner[k];
    // Generate K alternative parses using the K sampled parameters vectors.
    for (int i = 0; i < k; i++) {
      runners[i] = new DecoderRunner(inst, parameters[i], alternativeHeads[i]);
      runners[i].start();
    }
    // Wait for all decoder threads to complete
    for (int i = 0; i < k; i++) {
      try {
        runners[i].join();
      } catch (InterruptedException e) {
        e.printStackTrace();
      }
    }
  }

  class DecoderRunner extends Thread {
    DependencyInstance inst;

    Parameters parameters;

    int[] heads;

    public DecoderRunner(DependencyInstance inst, Parameters parameters, int[] heads) {
      super();
      this.inst = inst;
      this.parameters = parameters;
      this.heads = heads;
    }

    @Override
    public void run() {
      depParser.decode(inst, 1, parameters, heads);
    }
  }

  // /////////////////////////////////////////////////////////////////
  // Single threaded implementation of produce-K-Alternatives.
  // /////////////////////////////////////////////////////////////////
  void produceKAlternatives_ST(DependencyInstance inst, int[][] alternativeHeads) {
    // Generate K alternative parses using the K sampled parameters vectors.
    for (int i = 0; i < k; i++) {
      depParser.decode(inst, 1, parameters[i], alternativeHeads[i]);
    }
  }
}
