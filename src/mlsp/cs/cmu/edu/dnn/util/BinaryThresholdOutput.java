package mlsp.cs.cmu.edu.dnn.util;

public class BinaryThresholdOutput implements OutputAdapter {
  
  private double threshold = 0.5;

  @Override
  public Object getSmoothedPrediction(double[] networkOutput) {
    for(int i = 0; i < networkOutput.length; i++) {
      double val = networkOutput[i];
      networkOutput[i] = val >= threshold ? 1 : 0;
    }
    return networkOutput;
  }

  @Override
  public boolean isCorrect(Object output, Object truth) {
    double[] out = (double[]) output;
    double[] t = (double[]) truth;
    for(int i = 0; i < out.length; i++) {
      if(out[i] != t[i]) {
        return false;
      }
    }
    return true;
  }

}
