package mlsp.cs.cmu.edu.dnn.util;

public class DefaultOutput implements OutputAdapter {

	@Override
	public Object getSmoothedPrediction(double[] networkOutput) {
		return networkOutput;
	}

}
