package mlsp.cs.cmu.edu.util;

public class CostFunction {

	public static double meanSqError(double output, double truth) {
		return 0.5 * Math.pow(output - truth, 2) ;
	}
	
	public static double meanSqErrorDerivative(double output, double truth) {
		return output - truth;
	}

}
