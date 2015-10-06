package mlsp.cs.cmu.edu.util;

public class CostFunction {

	public static double meanSqError(double output, double truth) {
		return 0.5 * Math.pow(output - truth, 2) ;
	}
	
	public static double meanSqErrorDerivative(double output, double truth) {
		return output - truth;
	}

	public static double meanSqError(double[] prediction, double[] truth) {
		double sum = 0;
		for(int i = 0; i < prediction.length; i++)
			sum += meanSqError(prediction[i], truth[i]);
		return sum;
	}
	
	public static double meanSqErrorDerivative(double[] prediction, double[] truth) {
		double sum = 0;
		for(int i = 0; i < prediction.length; i++)
			sum += meanSqErrorDerivative(prediction[i], truth[i]);
		return sum;
	}

}
