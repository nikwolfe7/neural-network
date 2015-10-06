package training;

public class DataInstance {
  
  private double[] input;
  private double[] output;
 
  /**
   * Assumes we have truth first from left to right, then input value 
   * 
   * @param inputDimension
   * @param outputDimension
   * @param vector
   */
  public DataInstance(int inputDimension, int outputDimension, double[] vector) {
    this.input = new double[inputDimension];
    this.output = new double[outputDimension];
    System.arraycopy(vector, outputDimension, input, 0, inputDimension);
    System.arraycopy(vector, 0, output, 0, outputDimension);
  }
  
  public double[] getInputVector() {
    return input;
  }
  
  public double[] getOutputTruthValue() {
    return output;
  }

}
