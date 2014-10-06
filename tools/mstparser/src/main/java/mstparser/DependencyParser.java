package mstparser;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

public class DependencyParser {

  public ParserOptions options;

  protected DependencyPipe pipe;

  private final DependencyDecoder decoder;

  protected Parameters params;

  Parameters getParams() {
    return params;
  }

  public DependencyParser(DependencyPipe pipe, ParserOptions options) {
    this.pipe = pipe;
    this.options = options;

    // Set up arrays
    params = new Parameters(pipe.dataAlphabet.size());
    decoder = options.secondOrder ? new DependencyDecoder2O(pipe) : new DependencyDecoder(pipe);
  }

  public void train(int[] instanceLengths, String trainfile, File train_forest) throws IOException {

    // System.out.print("About to train. ");
    // System.out.print("Num Feats: " + pipe.dataAlphabet.size());

    int i = 0;
    for (i = 0; i < options.numIters; i++) {

      System.out.print(" Iteration " + i);
      // System.out.println("========================");
      // System.out.println("Iteration: " + i);
      // System.out.println("========================");
      System.out.print("[");

      long start = System.currentTimeMillis();

      trainingIter(instanceLengths, trainfile, train_forest, i + 1);

      long end = System.currentTimeMillis();
      // System.out.println("Training iter took: " + (end-start));
      System.out.println("|Time:" + (end - start) + "]");
    }

    params.averageParams(i * instanceLengths.length);

  }

  private void trainingIter(int[] instanceLengths, String trainfile, File train_forest, int iter)
          throws IOException {

    int numUpd = 0;
    ObjectInputStream in = null;
    try {
      in = new ObjectInputStream(new FileInputStream(train_forest));
      boolean evaluateI = true;

      int numInstances = instanceLengths.length;

      for (int i = 0; i < numInstances; i++) {
        if ((i + 1) % 500 == 0) {
          System.out.print((i + 1) + ",");
          // System.out.println("  "+(i+1)+" instances");
        }

        int length = instanceLengths[i];

        // Get production crap.
        FeatureVector[][][] fvs = new FeatureVector[length][length][2];
        double[][][] probs = new double[length][length][2];
        FeatureVector[][][][] nt_fvs = new FeatureVector[length][pipe.types.length][2][2];
        double[][][][] nt_probs = new double[length][pipe.types.length][2][2];
        FeatureVector[][][] fvs_trips = new FeatureVector[length][length][length];
        double[][][] probs_trips = new double[length][length][length];
        FeatureVector[][][] fvs_sibs = new FeatureVector[length][length][2];
        double[][][] probs_sibs = new double[length][length][2];

        DependencyInstance inst;

        if (options.secondOrder) {
          inst = ((DependencyPipe2O) pipe).readInstance(in, length, fvs, probs, fvs_trips,
                  probs_trips, fvs_sibs, probs_sibs, nt_fvs, nt_probs, params);
        } else {
          inst = pipe.readInstance(in, length, fvs, probs, nt_fvs, nt_probs, params);
        }

        double upd = options.numIters * numInstances - (numInstances * (iter - 1) + (i + 1)) + 1;
        int K = options.trainK;
        Object[][] d = null;
        if (options.decodeType.equals("proj")) {
          if (options.secondOrder) {
            d = ((DependencyDecoder2O) decoder).decodeProjective(inst, fvs, probs, fvs_trips,
                    probs_trips, fvs_sibs, probs_sibs, nt_fvs, nt_probs, K);
          } else {
            d = decoder.decodeProjective(inst, fvs, probs, nt_fvs, nt_probs, K);
          }
        }
        if (options.decodeType.equals("non-proj")) {
          if (options.secondOrder) {
            d = ((DependencyDecoder2O) decoder).decodeNonProjective(inst, fvs, probs, fvs_trips,
                    probs_trips, fvs_sibs, probs_sibs, nt_fvs, nt_probs, K);
          } else {
            d = decoder.decodeNonProjective(inst, fvs, probs, nt_fvs, nt_probs, K);
          }
        }
        params.updateParamsMIRA(inst, d, upd);

      }

      // System.out.println("");
      // System.out.println("  "+numInstances+" instances");

      System.out.print(numInstances);
    } finally {
      Util.closeQuietly(in);
    }
  }

  // /////////////////////////////////////////////////////
  // Saving and loading models
  // /////////////////////////////////////////////////////
  public void saveModel(String file) throws IOException {
    ObjectOutputStream out = null;
    try {
      out = new ObjectOutputStream(new FileOutputStream(file));
      out.writeObject(params.parameters);
      out.writeObject(pipe.dataAlphabet);
      out.writeObject(pipe.typeAlphabet);
    } finally {
      Util.closeQuietly(out);
    }
  }

  public void loadModel(String file) throws Exception {
    InputStream in = null;
    try {
      in = new FileInputStream(file);
      loadModel(in);
    } finally {
      Util.closeQuietly(in);
    }
  }

  public void loadModel(InputStream inputStream) throws IOException {
    try {
      ObjectInputStream is = new ObjectInputStream(inputStream);
      params.parameters = (double[]) is.readObject();
      pipe.dataAlphabet = (Alphabet) is.readObject();
      pipe.typeAlphabet = (Alphabet) is.readObject();
      pipe.closeAlphabets();
    } catch (ClassNotFoundException e) {
      IOException e2 = new IOException("Unable to load model: " + e.getMessage());
      e2.initCause(e);
      throw e2;
    }
  }

  // ////////////////////////////////////////////////////
  // Get Best Parses ///////////////////////////////////
  // ////////////////////////////////////////////////////

  public List<DependencyInstance> getParses() throws IOException {
    List<DependencyInstance> allInstances = new ArrayList<DependencyInstance>();
    outputParses(allInstances, false);
    return allInstances;
  }

  public void outputParses() throws IOException {
    outputParses(null, true);
  }

  /**
   * Get the parses.
   * 
   * @param allInstances
   *          a list to which all parse results are written. Can be {@code null}.
   * @param writeOutput
   *          write output to file and log some messages to screen.
   */
  protected void outputParses(List<DependencyInstance> allInstances, boolean writeOutput)
          throws IOException {

    String tFile = options.testfile;
    String file = null;
    if (writeOutput) {
      file = options.outfile;
    }

    ConfidenceEstimator confEstimator = null;
    if (options.confidenceEstimator != null) {
      confEstimator = ConfidenceEstimator.resolveByName(options.confidenceEstimator, this);
      System.out.println("Applying confidence estimation: " + options.confidenceEstimator);
    }

    long start = System.currentTimeMillis();

    pipe.initInputFile(tFile);
    if (writeOutput) {
      pipe.initOutputFile(file);
    }

    if (writeOutput) {
      System.out.print("Processing Sentence: ");
    }
    DependencyInstance instance = pipe.nextInstance();
    int cnt = 0;
    while (instance != null) {
      cnt++;
      if (writeOutput) {
        System.out.print(cnt + " ");
      }
      String[] forms = instance.forms;
      String[] formsNoRoot = new String[forms.length - 1];
      String[] posNoRoot = new String[formsNoRoot.length];
      String[] cposNoRoot = new String[formsNoRoot.length];
      String[] labels = new String[formsNoRoot.length];
      int[] heads = new int[formsNoRoot.length];

      decode(instance, options.testK, params, formsNoRoot, cposNoRoot, posNoRoot, labels, heads, confEstimator, writeOutput);
/*
      DependencyInstance parsedInstance;
      if (confEstimator != null) {
        double[] confidenceScores = confEstimator.estimateConfidence(instance);
        parsedInstance = new DependencyInstance(formsNoRoot, posNoRoot, labels, heads,
                confidenceScores);
      } else {
        parsedInstance = new DependencyInstance(formsNoRoot, posNoRoot, labels, heads);
      }
      if (writeOutput) {
        pipe.outputInstance(parsedInstance);
      }
      if (allInstances != null) {
        allInstances.add(parsedInstance);
      }
*/
      // String line1 = ""; String line2 = ""; String line3 = ""; String line4 = "";
      // for(int j = 1; j < pos.length; j++) {
      // String[] trip = res[j-1].split("[\\|:]");
      // line1+= sent[j] + "\t"; line2 += pos[j] + "\t";
      // line4 += trip[0] + "\t"; line3 += pipe.types[Integer.parseInt(trip[2])] + "\t";
      // }
      // pred.write(line1.trim() + "\n" + line2.trim() + "\n"
      // + (pipe.labeled ? line3.trim() + "\n" : "")
      // + line4.trim() + "\n\n");

      instance = pipe.nextInstance();
    }
    pipe.close();

    if (writeOutput) {
      long end = System.currentTimeMillis();
      System.out.println("Took: " + (end - start));
    }

  }

  static double[] scores;
  static BufferedWriter scoreWriter;

  // ////////////////////////////////////////////////////
  // Decode single instance
  // ////////////////////////////////////////////////////
  String[] decode(DependencyInstance instance, int K, Parameters params) {
	//System.out.println(K);

    String[] forms = instance.forms;

    int length = forms.length;

    FeatureVector[][][] fvs = new FeatureVector[forms.length][forms.length][2];
    double[][][] probs = new double[forms.length][forms.length][2];
    FeatureVector[][][][] nt_fvs = new FeatureVector[forms.length][pipe.types.length][2][2];
    double[][][][] nt_probs = new double[forms.length][pipe.types.length][2][2];
    FeatureVector[][][] fvs_trips = new FeatureVector[length][length][length];
    double[][][] probs_trips = new double[length][length][length];
    FeatureVector[][][] fvs_sibs = new FeatureVector[length][length][2];
    double[][][] probs_sibs = new double[length][length][2];
    if (options.secondOrder) {
      ((DependencyPipe2O) pipe).fillFeatureVectors(instance, fvs, probs, fvs_trips, probs_trips,
              fvs_sibs, probs_sibs, nt_fvs, nt_probs, params);
    } else {
      pipe.fillFeatureVectors(instance, fvs, probs, nt_fvs, nt_probs, params);
    }

    Object[][] d = null;
    if (options.decodeType.equals("proj")) {
      if (options.secondOrder) {
        d = ((DependencyDecoder2O) decoder).decodeProjective(instance, fvs, probs, fvs_trips,
                probs_trips, fvs_sibs, probs_sibs, nt_fvs, nt_probs, K);
      } else {
        d = decoder.decodeProjective(instance, fvs, probs, nt_fvs, nt_probs, K);
      }
    }
    if (options.decodeType.equals("non-proj")) {
      if (options.secondOrder) {
        d = ((DependencyDecoder2O) decoder).decodeNonProjective(instance, fvs, probs, fvs_trips,
                probs_trips, fvs_sibs, probs_sibs, nt_fvs, nt_probs, K);
      } else {
        d = decoder.decodeNonProjective(instance, fvs, probs, nt_fvs, nt_probs, K);
      }
    }

	// print all resulting parses
    StringBuffer buff = new StringBuffer();
    scores = new double[d.length];
    for (int i = 0; i < d.length; i++) {
      buff.append((String) d[i][1]).append("\n");
      scores[i] = (Double) d[i][2];
    }

    // convert scores to log prob
    //double logSum = logSumOfExponentials(scores);
    //for (int i = 0; i < d.length; i++) {
    //  if (d[i][1] != null) 
    //    scores[i] = scores[i] - logSum;
    //} 

    String[] res = buff.toString().split("\n");
    return res;
  }

	public static double maximum(double[] xs) {
		double m = xs[0];
		for (int i = 1; i < xs.length; i++) {
			if (xs[i] > m) 
				m = xs[i];
		}
		return m;
	}
	
	public static double logSumOfExponentials(double[] xs) {
		if (xs.length == 1)
			return xs[0];
		double max = maximum(xs);
		double sum = 0.0;
		for (int i = 0; i < xs.length; ++i)
			if (xs[i] != Double.NEGATIVE_INFINITY)
				sum += java.lang.Math.exp(xs[i] - max);
		return max + java.lang.Math.log(sum);
	}

  public void decode(DependencyInstance instance, int K, Parameters params, String[] formsNoRoot,
          String[] cposNoRoot, String[] posNoRoot, String[] labels, int[] heads, ConfidenceEstimator confEstimator, boolean writeOutput) throws IOException {

    String[] results = decode(instance, K, params);
    
    int i = 0;
    while (i < results.length && !results[i].equals("null")) {
      // write scores
      scoreWriter.write(scores[i] + " ");

      // System.out.println(results[i]);
      String[] res = results[i].split(" ");
      String[] forms = instance.forms;
      String[] cpos = instance.cpostags;
      String[] pos = instance.postags;

      for (int j = 0; j < forms.length - 1; j++) {
        formsNoRoot[j] = forms[j + 1];
        cposNoRoot[j] = cpos[j+1];
        posNoRoot[j] = pos[j + 1];
        String[] trip = res[j].split("[\\|:]"); // System.out.println(res[j]);
        labels[j] = pipe.types[Integer.parseInt(trip[2])];
        heads[j] = Integer.parseInt(trip[0]);
      }
     
      DependencyInstance parsedInstance;
      if (confEstimator != null) {
        double[] confidenceScores = confEstimator.estimateConfidence(instance);
        parsedInstance = new DependencyInstance(formsNoRoot, cposNoRoot, posNoRoot, labels, heads,
                confidenceScores);
      } else {
        parsedInstance = new DependencyInstance(formsNoRoot, cposNoRoot, posNoRoot, labels, heads);
      }
      if (writeOutput) {
        pipe.outputInstance(parsedInstance);
      }

      i++;
    }
    scoreWriter.write("\n");
  }

  public void decode(DependencyInstance instance, int K, Parameters params, int[] heads) {

    String[] res = decode(instance, K, params);

    for (int j = 0; j < instance.forms.length - 1; j++) {
      String[] trip = res[j].split("[\\|:]");
      heads[j] = Integer.parseInt(trip[0]);
    }
  }

  // ///////////////////////////////////////////////////
  // RUNNING THE PARSER
  // //////////////////////////////////////////////////
  public static void main(String[] args) throws FileNotFoundException, Exception {

    ParserOptions options = new ParserOptions(args);

    if (options.train) {

      DependencyPipe pipe = options.secondOrder ? new DependencyPipe2O(options)
              : new DependencyPipe(options);

      int[] instanceLengths = pipe.createInstances(options.trainfile, options.trainforest);

      pipe.closeAlphabets();

      DependencyParser dp = new DependencyParser(pipe, options);

      int numFeats = pipe.dataAlphabet.size();
      int numTypes = pipe.typeAlphabet.size();
      System.out.print("Num Feats: " + numFeats);
      System.out.println(".\tNum Edge Labels: " + numTypes);

      dp.train(instanceLengths, options.trainfile, options.trainforest);

      System.out.print("Saving model...");
      dp.saveModel(options.modelName);
      System.out.print("done.");

    }

    if (options.test) {
      DependencyPipe pipe = options.secondOrder ? new DependencyPipe2O(options)
              : new DependencyPipe(options);
      scoreWriter = new BufferedWriter(new FileWriter(options.outfile + ".mstscores"));

      DependencyParser dp = new DependencyParser(pipe, options);

      System.out.print("\tLoading model...");
      dp.loadModel(options.modelName);
      System.out.println("done.");

      pipe.closeAlphabets();

      dp.outputParses();
      scoreWriter.close();
    }

    System.out.println();

    if (options.eval) {
      System.out.println("\nEVALUATION PERFORMANCE:");
      DependencyEvaluator.evaluate(options.goldfile, options.outfile, options.format,
              (options.confidenceEstimator != null));
    }

    if (options.rankEdgesByConfidence) {
      System.out.println("\nRank edges by confidence:");
      EdgeRankerByConfidence edgeRanker = new EdgeRankerByConfidence();
      edgeRanker.rankEdgesByConfidence(options.goldfile, options.outfile, options.format);
    }
  }

}
