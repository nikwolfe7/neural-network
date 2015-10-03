package mlsp.cs.cmu.edu.elements;

/**
 * Output based on derivative of mean
 * squared error: 0.5 * (O - T) ^ 2
 * 
 * @author nwolfe
 */
public class Output extends Neuron {

  private volatile double outputTruthValue = 0;
  
  @Override
  public void backward() {
    /* output derivative is (O - T) */
    setErrorTerm((getOutput() - outputTruthValue) * derivative());
  }

  public void setTruthValue(double val) {
    outputTruthValue = val;
  }

}
