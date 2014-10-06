package mstparser;

public abstract class ConfidenceEstimator {

  public static ConfidenceEstimator resolveByName(String confEstimatorName,
          DependencyParser depParser) {

    // Expected name format: Type*param1*param2*...
    String[] params = confEstimatorName.split("\\*");
    String type = params[0];

    if (type.equals("KDFix")) {
      // Expected format: KDFix*stddev*K
      double stddev = Double.parseDouble(params[1]);
      int K = Integer.parseInt(params[2]);
      return new ConfidenceEstimatorKDFix(stddev, K, depParser);
    }

    throw new RuntimeException("Unknown confidence estimator: " + confEstimatorName);
  }

  // Compute the confidence score for each predicted edge.
  public abstract double[] estimateConfidence(DependencyInstance inst);
}