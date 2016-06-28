package mlsp.cs.cmu.edu.dnn.util;

public class CostFunction {

	/**
	 * Mean squared error cost function
	 * 
	 * E = 1/2 (o - y) ^ 2
	 * 
	 * @param output
	 * @param truth
	 * @return the squared error
	 */
	public static double meanSqError(double output, double truth) {
		return 0.5 * Math.pow(output - truth, 2);
	}

	/**
	 * Cross entropy cost function
	 * 
	 * E = -(y log(o) + (1 - y) log(1 - o))
	 * 
	 * @param output
	 * @param truth
	 * @return the cross entropy between the output and the truth
	 */
	public static double crossEntropy(double output, double truth) {
		return -(truth * Math.log(output)) - ((1 - truth) * Math.log(1 - output));
		//return -(truth * Math.log(output));
	}

	public static double crossEntropyDerivative(double output, double truth) {
		return output - truth;
	}

	public static double meanSqErrorDerivative(double output, double truth) {
		return output - truth;
	}

	public static double meanSqError(double[] prediction, double[] truth) {
		double sum = 0;
		for (int i = 0; i < prediction.length; i++)
			sum += meanSqError(prediction[i], truth[i]);
		return sum;
	}

	public static double crossEntropy(double[] prediction, double[] truth) {
		double sum = 0;
		for (int i = 0; i < prediction.length; i++)
			sum += crossEntropy(prediction[i], truth[i]);
		return sum;
	}

}
