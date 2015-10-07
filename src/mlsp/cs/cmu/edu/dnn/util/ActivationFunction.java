package mlsp.cs.cmu.edu.dnn.util;

public class ActivationFunction {

	public static double sigmoid(double input) {
		return 1.0 / (1.0 + Math.exp(-input));
	}

	public static double linear(double input) {
		return input;
	}

	public static double tanh(double input) {
//		return 1.0 * Math.tanh(input);
		 return 1.7159 * Math.tanh(0.66666666666 * input);
	}

	public static double sigmoidDerivative(double output) {
		return output * (1 - output);
	}

	public static double linearDerivative(double output) {
		return output;
	}

	public static double tanhDerivative(double output) {
//		return 1.0 / Math.pow(Math.cosh(output), 2);
		 return 1.14393333333 / Math.pow(Math.cosh(0.66666666666 * output), 2);
	}

}
