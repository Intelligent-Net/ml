import java.io.*;
import java.util.*;
import java.util.stream.*;

public class ReadCSV
{
  private Map<Integer,String> headers = new TreeMap<>();
  private double[] weights; // for LogReg version
  private double[] average; // for Anomaly version
  private double[] stddev; // for Anomaly version
  private String labelName = "Class";
  private int randomSeed = -1;
  private int sets;
  private int folds;
  private int trainingIterations = 10000;
  private double subSampleSize = -1.0;
  private double subSampleLabel = 0.0;
  private double corrThreshold = 0.7;
  private double apportionRatio = 0.8;
  private double f1Threshold;
  private double rate = 0.01;
  private Random rand;
  private boolean anomaly = false;

  public static void main(String[] args)
  {
    if (args.length != 1)
    {
      System.err.println("Format : ReadCSV <csv file name>");
      System.err.println("Meta parameters set as properties");

      System.exit(1);
    }

    ReadCSV model = new ReadCSV();

    model.anomaly = Boolean.parseBoolean(System.getProperty("anomaly", model.anomaly + ""));

    // Set threshold defaults
    if (model.anomaly)
    {
      model.sets = 1;
      model.folds = 1;
      model.f1Threshold = 1.0e-20;
    }
    else
    {
      model.sets = 20;
      model.folds = 5;
      model.f1Threshold = 0.998;
    }

    model.labelName = System.getProperty("Label", model.labelName);
    model.randomSeed = Integer.parseInt(System.getProperty("Seed", model.randomSeed + ""));
    model.sets = Integer.parseInt(System.getProperty("sets", model.sets + ""));
    model.folds = Integer.parseInt(System.getProperty("folds", model.folds + ""));
    model.trainingIterations = Integer.parseInt(System.getProperty("trainingIterations", model.trainingIterations + ""));
    model.apportionRatio = Double.parseDouble(System.getProperty("apportionRatio", model.apportionRatio + ""));
    model.subSampleSize = Double.parseDouble(System.getProperty("subSampleSize", model.subSampleSize + ""));
    model.subSampleLabel = Double.parseDouble(System.getProperty("subSampleLabel", model.subSampleLabel + ""));
    model.corrThreshold = Double.parseDouble(System.getProperty("corrThreshold", model.corrThreshold + ""));
    model.f1Threshold = Double.parseDouble(System.getProperty("f1Threshold", model.f1Threshold + ""));
    model.rate = Double.parseDouble(System.getProperty("rate", model.rate + ""));
    model.rand = model.randomSeed >= 0 ? new Random(model.randomSeed) : new Random();

    model.modeller(args);
  }

  public void modeller(String... fn)
  {
    // Load original data from given file
    List<Map<String,Double>> allData = load(fn, "Time");

    System.out.println("Rows Loaded : " + allData.size());

    // We can select a subset of data if we know something about a feature..
    //allData = pruneAttValue(allData, "Amount", 0.0, 1.0, true);

    //System.out.println("Rows Remaining : " + allData.size());

    // Amount should be normalised
    //allData = normalise(allData, "Amount");
    //allData = normalise(allData);

    double[] labels = new double[allData.size()];
    double[][] atts = new double[headers.size() - 1][allData.size()];
    String[] heads = new String[headers.size() - 1];

    flatten(allData, heads, labels, atts, true);
//System.err.println(Arrays.toString(heads));
//System.err.println(atts.length);
//System.err.println(heads.length);

    // How statistically significant are features
    if (corrThreshold > 0.0)
    {
      Map<Integer,String> newHeads = significance(atts, labels, heads, 1.0);
System.err.println(newHeads);

      keep(allData, newHeads.values().toArray(new String[0]));

      if (newHeads.size() > 2)
        headers = newHeads;
    }

    //allData = normalise(allData);

    // split data up for test training and testing
    List<Map<String,Double>>[] landT = apportion(allData, apportionRatio);
    List<Map<String,Double>> data = landT[0];
    List<Map<String,Double>> test = landT[1];

    System.out.println("Data : " + data.size());
    System.out.println("Test : " + test.size());

    /*
    newHeads = newHeaders("V1", "V10", "V11", "V12", "V14", "V16", "V17", "V18", "V2", "V3", "V4", "V5", "V6", "V7", "V9", "V1_V3", "V2_V4", "V2_V1", "V2_V2");

    keep(data, newHeads.values().toArray(new String[0]));
    */
    //addHigherOrder(data, "V1_V3", "V2_V4", "V2_V1", "V2_V2");
    //addHigherOrder(data, "V1_V1", "V2_V2", "V3_V3", "V4_V4", "V5_V5", "V6_V6", "V7_V7", "V9_V9", "V10_V10", "V11_V11", "V12_V12", "V14_V14", "V16_V16", "V17_V17", "V18_V18");
    //secondOrder(data);

    /*

    labels = new double[data.size()];
    atts = new double[headers.size() - 1][data.size()];
    heads = new String[headers.size() - 1];

    flatten(data, heads, labels, atts, true);

    newHeads = significance(atts, labels, heads, 0.6, 1.0);
System.err.println("2 = " + newHeads.values());

    keep(data, newHeads.values().toArray(new String[0]));

    headers = newHeads;
    */

    double[] bestWeights = null;
    double lastF1 = 0.0;

    heads = new String[headers.size() - 1];
System.err.println(heads.length);

    // As data unbalanced, use several random subsets
    for (int i = 0; i != sets; i++)
    {
      // Get a subsample of the data (all frauds, and balanced set of non-fraud)
      List<Map<String,Double>> sample = anomaly ? data : subSample(data, subSampleLabel, subSampleSize);

      // N fold validation
      for (int j = 0; j != folds; j++)
      {
        // Split up for training and validation data folds
        List<Map<String,Double>> train = null;
        List<Map<String,Double>> validate = null;

        if (anomaly)
        {
          // Can remove frauds but makes little difference, as relatively small!
          train = sample;
          //train = pruneAttValue(sample, labelName, 1.0, true);
        }
        else
        {
          int cnt = 0;

          do
          {
            List<Map<String,Double>>[] data2 = apportion(sample, apportionRatio);

            train = data2[0];
            validate = data2[1];

            if (cnt++ > 3)
            {
              System.err.println("Data set size is to small to learn");

              System.exit(1);
            }
          }
          while (train.size() < headers.size());
//System.out.println(data2[0].size() + " + " + data2[1].size() + " = " + sample.size());
        }

        if (anomaly)
          atts = new double[train.size()][headers.size() - 1];
        else
          atts = new double[headers.size() - 1][train.size()];

        labels = new double[train.size()];

        flatten(train, heads, labels, atts, ! anomaly);

//System.err.println("-------");
//System.err.println(train.size() + " - " + atts.length + " : " +  labels.length);
        weights = anomaly ? trainAnomaly(atts, labels) : train(trainingIterations, atts, labels);

        //System.out.println(Arrays.toString(heads));
        //System.out.println(Arrays.toString(weights));
        //System.out.println(data.get(0));

        if (! anomaly)
        {
          double f1 = evaluate(validate);

          if (f1 > lastF1)
          {
            lastF1 = f1;
            bestWeights = weights;
          }
        }
      }
    }

    if (! anomaly)
    {
      System.out.println();
      System.out.println("Best F1 = " + lastF1);
      System.out.println(Arrays.toString(bestWeights));
    }

    weights = bestWeights;

    System.out.println();
    //System.out.println("Evaluate against all data");
    double f1all = evaluate(test, true);

    /*
    System.out.println();
    System.out.println("Evaluate against frauds");
    double f1fraud = evaluate(test, 1.0, true);
    */

    //System.out.printf("Test F1 = %.1f %%\n", f1 * 100);
  }

  private double evaluate(List<Map<String,Double>> valData)
  {
    return evaluate(valData, -1.0, false);
  }

  private double evaluate(List<Map<String,Double>> valData, double label)
  {
    return evaluate(valData, label, false);
  }

  private double evaluate(List<Map<String,Double>> valData, boolean show)
  {
    return evaluate(valData, -1.0, show);
  }

  private double evaluate(List<Map<String,Double>> valData, double label, boolean show)
  {
    double[] cLabels = new double[valData.size()];
    double[][] cAtts = new double[valData.size()][headers.size() - 1];
    String[] cHeads = new String[headers.size() - 1];

    flatten(valData, cHeads, cLabels, cAtts, false);

    int count = 0;
    double[][] confusionMatrix = new double[2][2];
    double accuracy = 0.0;
    double precision = 0.0;
    double recall = 0.0;

    for (int k = 0; k != cLabels.length; k++)
    {
      double thisLabel = valData.get(k).get(labelName);

      if (label < 0.0 || thisLabel == label)
      {
        double predicted = anomaly ? classifyAnomaly(average, stddev, cAtts[k], false) : classify(weights, cAtts[k], false);

        if (anomaly ? predicted < f1Threshold : predicted > f1Threshold)
        {
          if (thisLabel == 1.0)
          {
            accuracy += 1;
            precision += 1;
            recall += 1;
            confusionMatrix[0][0] += 1;
          }
          else
          {
            confusionMatrix[1][0] += 1;
          }
        }
        else
        {
          if (valData.get(k).get(labelName) == 1.0)
          {
            confusionMatrix[0][1] += 1;
          }
          else
          {
            accuracy += 1;
            confusionMatrix[1][1] += 1;
          }
        }

        count++;
      }
    }

    accuracy /= count;
    precision /= confusionMatrix[0][0] + confusionMatrix[1][0];
    recall /= confusionMatrix[0][0] + confusionMatrix[0][1];

    double f1 = 2.0 * precision * recall / (precision + recall);

    if (show)
    {
      System.out.println();
      System.out.println(Arrays.deepToString(confusionMatrix));
      System.out.printf("Accuracy  : %.1f %%\n", accuracy * 100);
      System.out.printf("Precision : %.1f %%\n", precision * 100);
      System.out.printf("Recall    : %.1f %%\n", recall * 100);
      System.out.printf("F1        : %.1f %%\n", f1 * 100);
    }

    return f1;
  }

  private static double pearsonCorrelation(double[] xn, double[] yn)
  {
    return pearsonCorrelation(xn, yn, null, 0.0);
  }

  private static double pearsonCorrelation(double[] xn, double[] yn, double[] labels, double label)
  {
    long n = xn.length;        
    double xSum = 0;
    double ySum = 0;
    double xySum = 0;
    double xxSum = 0;
    double yySum = 0;
    
    for (int i = 0; i != n; i++)
    {
      if (labels == null || labels[i] == label)
      {
        double x = xn[i];
        double y = yn[i];

        if (x == 0.0 && label == 0.0)
          x = -1.0;
        if (y == 0.0 && label == 0.0)
          y = -1.0;

        xSum += x;
        ySum += y;
        xySum += x * y;
        xxSum += x * x;
        yySum += y * y;
      }
    }

    double xd = Math.sqrt(n * xxSum - xSum * xSum);
    double yd = Math.sqrt(n * yySum - ySum * ySum);

    if (xd == 0.0)
      xd = 1.0;
    if (yd == 0.0)
      yd = 1.0;
    
    return (n * xySum - xSum * ySum) / (xd * yd);
  }

  public double[] trainAnomaly(double[][] data, double[] labels)
  {
    this.average = new double[data[0].length];
    this.stddev = new double[data[0].length];

    for (int i = 0; i < average.length; i++)
    {
      double[] values = data[i];
      // possible transformation
//System.err.println(i + " - " + headers.get(i));
      if (false)
      {
//System.err.println(values[0] + " - " + headers.get(i));
        values = Arrays.stream(data[i])
                       .map(x -> - Math.log(x))
                       .toArray();
//System.err.println(values[i] + " - " + headers.get(i));
      }
      double average = Arrays.stream(values)
                             .summaryStatistics()
                             .getAverage();
      double variance = Arrays.stream(values)
                              .map(x -> Math.pow(x - average, 2.0))
                              .sum() / values.length;
      double stddev = Math.sqrt(variance);

      double[] sData = Arrays.copyOf(values, values.length);

      Arrays.sort(sData);

      double median = sData[sData.length / 2];
//System.err.println(headers.get(i) + " = " + average + " : " + stddev + " : " + median);
      this.average[i] = average;
      this.stddev[i] = stddev;
    }

    return weights;
  }

  private static double classifyAnomaly(double[] average, double[] stddev, double[] x, boolean debug)
  {
    double prob = 1.0;

    for (int i = 0; i < average.length; i++)
    {
      prob *= (1.0 / Math.sqrt(Math.pow(2.0 * Math.PI, stddev[i]))) * Math.exp(- (Math.pow(x[i] - average[i], 2.0) / Math.pow(2.0 * stddev[i], 2.0)));
    }

    if (debug)
      System.out.println(prob);

    return prob;
  }

  public double[] train(int its, double[][] data, double[] labels)
  {
    int m = data.length;
    double[] weights = new double[m - 1];
//System.err.println(Arrays.toString(labels));

    for (int n = 0; n < its; n++)
    {
      for (int i = 0; i < m; i++)
      {
        double[] x = data[i];
//System.err.println(m + " --- " + data[0].length);
        double predicted = classify(weights, x);
        double label = labels[i];

        for (int j = 0; j < weights.length; j++)
        {
          // Conspicuous by it's absence is regularisation!
          weights[j] = weights[j] + rate * (label - predicted) * x[j];
        }
      }
    }

    return weights;
  }

  private static double classify(double[] weights, double[] x)
  {
    return classify(weights, x, false);
  }

  private static double classify(double[] weights, double[] x, boolean debug)
  {
    double logit = 0.0;

//System.err.println(weights.length + " : " + x.length);
//System.err.println(Arrays.toString(weights));
//System.err.println(Arrays.toString(x));
    for (int i = 0; i < weights.length; i++)
      logit += weights[i] * x[i];

    if (debug)
      //System.out.println(logit + " = " + (1.0 / (1.0 + Math.exp(-logit))) + " : " + Arrays.toString(x));
      System.out.println((1.0 / (1.0 + Math.exp(-logit))));

    // sigmoid
    return 1.0 / (1.0 + Math.exp(-logit));
  }

  @SuppressWarnings("unchecked")
  public List<Map<String,Double>>[] apportion(Collection<Map<String,Double>> arr, double ratio)
  {
    List<Map<String,Double>>[] split = (ArrayList<Map<String,Double>>[]) new ArrayList[2];
    int l = arr.size();
    int n = (int) (l * ratio);
    List<Map<String,Double>> one = new ArrayList<>(n + 1);
    List<Map<String,Double>> two = new ArrayList<>(l - n + 1);

    split[0] = one;
    split[1] = two;

    for (Map<String,Double> i : arr)
    {
      if (rand.nextDouble() < ratio)
        one.add(i);
      else
        two.add(i);
    }

    return split;
  }

  private List<Map<String,Double>> subSample(Collection<Map<String,Double>> arr)
  {
    return subSample(arr, 0.0, -1.0);
  }

  private List<Map<String,Double>> subSample(Collection<Map<String,Double>> arr, double label)
  {
    return subSample(arr, label, -1.0);
  }

  private List<Map<String,Double>> subSample(Collection<Map<String,Double>> arr, double label, double sampleSize)
  {
    // not given so estimate from data
    if (sampleSize <= 0.0)
    {
      int labeled = 0;

      for (Map<String,Double> ents : arr)
      {
        if (ents.get(labelName) == label)
          labeled++;
      }

      sampleSize = 1.0 - (double) labeled / arr.size();
    }

    List<Map<String,Double>> sample = new ArrayList<Map<String,Double>>((int) (arr.size() * sampleSize * 2.0));
    int in = 0;
    int out = 0;

    for (Map<String,Double> ents : arr)
    {
      double clazz = ents.get(labelName);

      if (clazz == label)
      {
        if (rand.nextDouble() < sampleSize)
        {
          sample.add(ents);

          in++;
        }
      }
      else
      {
        sample.add(ents);

        out++;
      }
    }

    //System.err.println(in + " + " + out + " == " + sample.size());

    return sample;
  }

  private void flatten(List<Map<String,Double>> data, String[] hds, double[] labels, double[][] atts, boolean byRow)
  {
    int row = 0;

    for (Map<String,Double> ents : data)
    {
      flatten(row, ents, hds, labels, atts, byRow);

//System.err.println(Arrays.toString(atts));
//System.err.println(row);
      row++;
    }
  }

  private void flatten(Map<String,Double> data, String[] hds, double[] labels, double[][] atts, boolean byRow)
  {
    flatten(0, data, hds, labels, atts, byRow);
  }

  private void flatten(int row, Map<String,Double> data, String[] hds, double[] labels, double[][] atts, boolean byRow)
  {
    int i = 0;

    for (Map.Entry<String,Double> e : data.entrySet())
    {
      String key = e.getKey();

      if (labelName.equals(key))
      {
        labels[row] = e.getValue();
      }
      else
      {
        if (row == 0)
          hds[i] = key;

        if (byRow)
          atts[i][row] = e.getValue();
        else
          atts[row][i] = e.getValue();

        i++;
      }
    }
  }

  private Map<Integer,String> significance(double[][] atts, double[] labels, String[] heads)
  {
    return significance(atts, labels, heads, -1.0);
  }

  private Map<Integer,String> significance(double[][] atts, double[] labels, String[] heads, double label)
  {
    List<String> hds = new ArrayList<>();
    int i = 0;

    for (double[] at : atts)
    {
      double corr = pearsonCorrelation(at, labels, label >= 0.0 ? labels : null, 1.0);

      if (Math.abs(corr) > corrThreshold)
      {
        //System.out.println(i + " : " + heads[i] + " = " + corr);

        hds.add(heads[i]);
      }

      i++;
    }

    hds.add(labelName);  // Add in the Label
//System.err.println(Arrays.toString(heads));
//System.err.println(hds);

    Map<Integer,String> hm = new TreeMap<>();

    for (int j = 0; j != hds.size(); j++)
      hm.put(j, hds.get(j));

    return hm;
  }

  private Map<Integer,String> newHeaders(String... hdrs)
  {
    int i = 0;
    Map<Integer,String> h = new TreeMap<>();

    for (String hdr : hdrs)
    {
      h.put(i, hdr);

      i++;
    }

    return h;
  }

  private void remove(List<Map<String,Double>> data, String... hds)
  {
    for (Map<String,Double> ents : data)
    {
      for (String hdr : hds)
        ents.remove(hdr);
    }
  }

  private void keep(List<Map<String,Double>> data, String... hds)
  {
    List<String> rms = new ArrayList<>(headers.values());

    rms.removeAll(Arrays.asList(hds));

    remove(data, rms.toArray(new String[0]));
  }

  private List<Map<String,Double>> pruneAttValue(List<Map<String,Double>> data, String att, double value, boolean remove)
  {
    return pruneAttValue(data, att, value, value, remove);
  }

  private List<Map<String,Double>> pruneAttValue(List<Map<String,Double>> data, String att, double lVal, double hVal, boolean remove)
  {
    List<Map<String,Double>> newData = new ArrayList<>(data.size());

    for (Map<String,Double> ents : data)
    {
      double thisValue = ents.get(att);

      if (remove)
      {
        if (thisValue < lVal || thisValue > hVal)
          newData.add(ents);
      }
      else
      {
        if (thisValue >= lVal && thisValue <= hVal)
          newData.add(ents);
      }
    }

    return newData;
  }

  private void addHigherOrder(List<Map<String,Double>> data, String... hds)
  {
    boolean first = true;
    int i = headers.size();

    for (Map<String,Double> ents : data)
    {
      for (String hdr : hds)
      {
        String bits[] = hdr.split("_");
        Double v1 = ents.get(bits[0]);
        Double v2 = ents.get(bits[1]);

        if (v1 != null && v2 != null)
        {
          ents.put(hdr, v1 * v2);

          if (first)
          {
            headers.put(i, hdr);

            i++;
          }
        }
      }

      first = false;
    }
  }

  private void secondOrder(List<Map<String,Double>> data)
  {
    String[] vals = headers.values().toArray(new String[0]);

    for (Map<String,Double> ents : data)
    {
      for (String hdr1 : vals)
      {
        if (hdr1.equals(labelName))
          continue;

        for (String hdr2 : vals)
        {
          if (hdr2.equals(labelName))
            continue;

          String hdr = hdr1 + "_" + hdr2;
//System.err.println(i + " = " + hdr);

          ents.put(hdr, ents.get(hdr1) * ents.get(hdr2));
        }
      }
//System.exit(0);
    }

    int i = headers.size();

    for (String hdr1 : vals)
    {
      if (hdr1.equals(labelName))
        continue;

      for (String hdr2 : vals)
      {
        if (hdr2.equals(labelName))
          continue;

        String hdr = hdr1 + "_" + hdr2;

        headers.put(i, hdr);

        i++;
      }
    }
//System.err.println(headers);
  }

  private List<Map<String,Double>> normalise(List<Map<String,Double>> data, String... hds)
  {
    Collection<String> heads = hds.length == 0 ? new ArrayList<String>(data.get(0).keySet()) : Arrays.asList(hds);
    Map<String, Double> mins = new HashMap<>();
    Map<String, Double> maxs = new HashMap<>();

    heads.remove(labelName);

    for (Map<String,Double> ents : data)
    {
      for (String hdr : heads)
      {
        Double amount = ents.get(hdr);

        if (amount != null)
        {
          if (mins.containsKey(hdr))
          {
            if (amount < mins.get(hdr))
              mins.replace(hdr, amount);
          }
          else
            mins.put(hdr, amount);

          if (maxs.containsKey(hdr))
          {
            if (amount > maxs.get(hdr))
              maxs.replace(hdr, amount);
          }
          else
            maxs.put(hdr, amount);
        }
      }
    }

    for (Map<String,Double> ents : data)
    {
      for (String hdr : heads)
      {
        Double amount = ents.get(hdr);

        if (amount != null)
        {
          double maxv = maxs.get(hdr);
          double minv = mins.get(hdr);
          double val = (amount - minv) / (maxv - minv);

//System.err.println(hdr + " : (" + amount + " - " + minv + ") / (" + maxv + " - " + minv + ") = " + val);
          ents.replace(hdr, val);
        }
      }
    }

    return data;
  }

  private List<Map<String,Double>> load(String[] fns, String... excludes)
  {
    List<Map<String,Double>> data = new LinkedList<>();

    Set<String> exHead = new HashSet<>();
    Set<Integer> exPos = new HashSet<>();

    for (String hd : excludes)
      exHead.add(hd);

    for (String fn : fns)
    {
      System.out.println("Loading : " + fn);

      String cvsSplitBy = fn.endsWith(".txt") ? "\t" : ",";

      try (BufferedReader br = new BufferedReader(new FileReader(fn)))
      {
        boolean first = true;
        String line;

        while ((line = br.readLine()) != null)
        {
          if (first)
          {
            int i = 0;

            for (String hdr : line.split(cvsSplitBy))
            {
              if (hdr.startsWith("\"") && hdr.endsWith("\"") || hdr.startsWith("'") && hdr.endsWith("'"))
                hdr = hdr.substring(1, hdr.length() - 1);

              if (exHead.contains(hdr))
              {
                exPos.add(i);
              }
              else
              {
                headers.put(i, hdr);
              }

              i++;
            }

            first = false;

            continue;
          }

          // use comma as separator
          String[] items = line.split(cvsSplitBy);
          Map<String,Double> its = new TreeMap<>();

          try
          {
            int i = 0;

            for (String it : line.split(cvsSplitBy))
            {
              if (it.startsWith("\"") && it.endsWith("\"") || it.startsWith("'") && it.endsWith("'"))
                it = it.substring(1, it.length() - 1);

              try
              {
                if (! exPos.contains(i))
                  its.put(headers.get(i), Double.parseDouble(it));
              }
              catch (Exception e)
              {
                System.err.println(e);
              }

              i++;
            }

            data.add(its);
          }
          catch (ArrayIndexOutOfBoundsException ae)
          {
            System.err.println("Bad Record : " + line);
          }
        }
      }
      catch (Exception e)
      {
        e.printStackTrace();
      }
    }

    return data;
  }
}
