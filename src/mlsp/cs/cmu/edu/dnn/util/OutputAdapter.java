package mlsp.cs.cmu.edu.dnn.util;

public interface OutputAdapter {
	
	public Object getSmoothedPrediction(double[] networkOutput);

}
