package mlsp.cs.cmu.edu.dnn.util;

public class MaxBinaryThresholdOutput extends BinaryThresholdOutput {

	@Override
	public Object getSmoothedPrediction(double[] networkOutput) {
		double max = Double.MIN_VALUE;
		for (int i = 0; i < networkOutput.length; i++) 
			max = (networkOutput[i] > max) ? networkOutput[i] : max;
		for (int i = 0; i < networkOutput.length; i++)
			networkOutput[i] = (networkOutput[i] == max) ? 1.0 : 0.0;
		return networkOutput;
	}

}
