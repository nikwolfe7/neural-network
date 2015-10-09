package mlsp.cs.cmu.edu.dnn.util;

public class DefaultOutput implements OutputAdapter {

	@Override
	public Object getSmoothedPrediction(double[] networkOutput) {
		return networkOutput;
	}

	/**
	 * Checks for direct equivalence between two double arrays
	 */
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
