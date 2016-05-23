package mlsp.cs.cmu.edu.dnn.util;

public class ActivationFunction {

	/***
	 * FUNCTIONS 
	 */
	
	public static double sigmoid(double input) {
		return 1.0 / (1.0 + Math.exp(-input));
	}

	public static double exp(double input) {
	  return Math.exp(input);
	}

	public static double tanh(double input) {
//		return Math.tanh(input);
		 return 1.7159 * Math.tanh(0.66666666666 * input);
	}
	
	public static double relu(double input) {
		return Math.max(0, input);
	}
	
	public static double leakyRelu(double input) {
		return (input > 0) ? input : 0.01 * input;
	}
	
	/* Relu approximation... */
	public static double softPlus(double input) {
		return Math.log(1 + Math.exp(input));
	}

	
	/***
	 * DERIVATIVES 
	 */
	
	public static double sigmoidDerivative(double output) {
		return output * (1 - output);
	}
	
	 public static double sigmoidSecondDerivative(double output) {
	    /* s'(x)*(1 - 2*s(x)) */
	    return sigmoidDerivative(output) * (1 - 2 * output);
	  }

	public static double expDerivative(double output) {
	  return output;
	}

	public static double tanhDerivative(double output) {
//		return 1.0 - Math.pow(output, 2);
		 return 1.14393333333 - Math.pow((0.66666666666 * output), 2);
	}
	
	/* Calling functions must pass the input to the neuron here */
	public static double reluDerivative(double input) {
		return (input > 0) ? 1 : 0; 
	}

	/* Calling functions must pass the input to the neuron here */
	public static double leakyReluDerivative(double input) {
		return (input > 0) ? 1 : 0.01; 
	}
	
	/* Calling functions must pass the input to the neuron here */
	public static double softPlusDerivative(double input) {
		return sigmoid(input);
	}
 

}
