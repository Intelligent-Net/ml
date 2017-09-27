package uk.co.inet.card;

import java.io.*;
import java.util.*;
import java.util.stream.*;

public class CreditCardAnomaly
{
  private Map<Integer,String> headers = new TreeMap<>();
  private double[] average; // for Anomaly version
  private double[] stddev; // for Anomaly version
  private String labelName = "Class";
  private int randomSeed = -1;
  private double corrThreshold = 0.7;
  private double apportionRatio = 0.8;
  private double f1Threshold = 1.0e-21;
  private Random rand;

  public static void main(String[] args)
  {
    if (args.length != 1)
    {
      System.err.println("Format : CreditCardAnomaly <csv file name>");
      System.err.println("Meta parameters set as properties");

      System.exit(1);
    }

    CreditCardAnomaly model = new CreditCardAnomaly();

    model.labelName = System.getProperty("Label", model.labelName);
    model.randomSeed = Integer.parseInt(System.getProperty("Seed", model.randomSeed + ""));
    model.apportionRatio = Double.parseDouble(System.getProperty("apportionRatio", model.apportionRatio + ""));
    model.corrThreshold = Double.parseDouble(System.getProperty("corrThreshold", model.corrThreshold + ""));
    model.f1Threshold = Double.parseDouble(System.getProperty("f1Threshold", model.f1Threshold + ""));
    model.rand = model.randomSeed >= 0 ? new Random(model.randomSeed) : new Random();

    model.modeller(args);
  }

  public void modeller(String... fn)
  {
    // Load original data from given file
    List<Map<String,Double>> allData = load(fn, "Time");

    System.out.println("Rows Loaded : " + allData.size());

    // Amount should be normalised
    allData = normalise(allData, "Amount");

    double[] labels = new double[allData.size()];
    double[][] atts = new double[headers.size() - 1][allData.size()];
    String[] heads = new String[headers.size() - 1];

    flatten(allData, heads, labels, atts, true);

    // How statistically significant are features
    Map<Integer,String> newHeads = significance(atts, labels, heads, 1.0);

    System.err.println("Chosen fields : " + newHeads);

    keep(allData, newHeads.values().toArray(new String[0]));

    if (newHeads.size() > 2)
      headers = newHeads;

    // split data up for test training and testing
    List<Map<String,Double>>[] landT = apportion(allData, apportionRatio);
    List<Map<String,Double>> data = landT[0];
    List<Map<String,Double>> test = landT[1];

    System.out.println("Data : " + data.size());
    System.out.println("Test : " + test.size());

    heads = new String[headers.size() - 1];

    // Get a subsample of the data (all frauds, and balanced set of non-fraud)
    List<Map<String,Double>> sample = data;

    // Split up for training and validation data folds
    List<Map<String,Double>> train = null;
    List<Map<String,Double>> validate = null;

    // Can remove frauds but makes little difference, as relatively small!
    train = sample;

    labels = new double[train.size()];
    atts = new double[train.size()][headers.size() - 1];

    flatten(train, heads, labels, atts, false);

    trainAnomaly(atts, labels);

    System.out.println();
    double f1all = evaluate(test, true);
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
        double predicted = classifyAnomaly(average, stddev, cAtts[k], false);

        if (predicted < f1Threshold)
        {
          if (thisLabel == 1.0)
          {
            accuracy++;
            precision++;
            recall++;
            confusionMatrix[0][0]++;
          }
          else
          {
            confusionMatrix[0][1]++;
          }
        }
        else
        {
          if (thisLabel == 1.0)
          {
            confusionMatrix[1][0]++;
          }
          else
          {
            accuracy++;
            confusionMatrix[1][1]++;
          }
        }

        count++;
      }
    }

    accuracy /= count;
    precision /= precision + confusionMatrix[1][0];
    recall /= recall + confusionMatrix[0][1];

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

  public void trainAnomaly(double[][] data, double[] labels)
  {
    this.average = new double[data[0].length];
    this.stddev = new double[data[0].length];

    for (int i = 0; i < average.length; i++)
    {
      double[] values = data[i];
      // possible transformation
      /*
      double[] values = Arrays.stream(data[i])
                              .map(x -> Math.pow(x, 1.0))
                              .toArray();
      */
      double average = Arrays.stream(values)
                             .summaryStatistics()
                             .getAverage();
      double variance = Arrays.stream(values)
                              .map(x -> Math.pow(x - average, 2.0))
                              .sum() / values.length;
      double stddev = Math.sqrt(variance);

//      double[] sData = Arrays.copyOf(values, values.length);

//      Arrays.sort(sData);

//      double median = 0.0; //sData[sData.length / 2];
      this.average[i] = average;
      this.stddev[i] = stddev;
    }
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

  private void flatten(List<Map<String,Double>> data, String[] hds, double[] labels, double[][] atts, boolean byRow)
  {
    int row = 0;

    for (Map<String,Double> ents : data)
    {
      flatten(row, ents, hds, labels, atts, byRow);

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

  private static List<Map<String,Double>> normalise(List<Map<String,Double>> data, String... hds)
  {
    Map<String, Double> sums = new HashMap<>();

    for (Map<String,Double> ents : data)
    {
      for (String hdr : hds)
      {
        Double amount = ents.get(hdr);

        if (amount != null)
        {
          if (sums.containsKey(hdr))
            sums.replace(hdr, amount + sums.get(hdr));
          else
            sums.put(hdr, amount);
        }
      }
    }

    for (Map<String,Double> ents : data)
    {
      for (String hdr : hds)
      {
        Double amount = ents.get(hdr);

        if (amount != null)
        {
          double total = sums.get(hdr);

          ents.replace(hdr, amount / total);
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
