package mlsp.cs.cmu.edu.dnn.util;

public interface OutputAdapter {
	
	public Object getSmoothedPrediction(double[] networkOutput);
	
	public boolean isCorrect(Object output, Object truth);

}
