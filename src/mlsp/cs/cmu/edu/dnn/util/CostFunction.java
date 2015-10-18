package mlsp.cs.cmu.edu.dnn.util;

public class CostFunction {

	public static double meanSqError(double output, double truth) {
		return 0.5 * Math.pow(output - truth, 2) ;
	}
	
	public static double crossEntropy(double output, double truth) {
	  return -1 * Math.log(output) * truth;
 	}
	
	public static double crossEntropyDerivative(double output, double truth) {
	  return output - truth;
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
	
	public static double crossEntropy(double[] prediction, double[] truth) {
		double sum = 0;
		for (int i = 0; i < prediction.length; i++)
			sum += crossEntropy(prediction[i], truth[i]);
		return sum;
	}

}
