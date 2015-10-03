package mlsp.cs.cmu.edu.util;

public class ActivationFunction {
  
  /*================================================*/
  /* Public functions to change
  /*================================================*/
  public static double getActivation(double input) {
      return sigmoid(input);
  }
  
  public static double getActivationAndCalculateDerivative(double input) {
    return getSigmoidActivationAndCalculateDerivative(input);
  }
  
  public static double getActivationDerivative(double output) {
    return getSigmoidActivationDerivative(output);
  }
  
  /*================================================*/
  /* Private functions
  /*================================================*/
  private static double sigmoid(double input) {
    return 1.0 / (1.0 + Math.exp(-input));
  }
  
  private static double getSigmoidActivationAndCalculateDerivative(double input) {
    return sigmoid(input) * (1 - sigmoid(input));
  }

  private static double getSigmoidActivationDerivative(double output) {
    return output * (1 - output);
  }

  public static void main(String[] args) {
    double i = -5;
    while (i < 5) {
      System.out.println(i + "," + ActivationFunction.getActivation(i));
      i += 0.1;
    }
  }

}
