package mlsp.cs.cmu.edu.elements;

/**
 * Output based on derivative of mean
 * squared error: 0.5 * (T - O) ^ 2
 * 
 * @author nwolfe
 */
public class Output extends Neuron {

  private volatile double outputTruthValue = 0;

  @Override
  public void forward() {
    double sum = 0;
    for (NetworkElement e : getIncomingElements())
      sum += e.getOutput();
    setOutput(sum);
  }

  @Override
  public void backward() {
    setErrorTerm(derivative());
  }

  @Override
  public double derivative() {
    /* output derivative is (T - O) */
    return getOutput() - outputTruthValue;
  }

  public void setTruthValue(double val) {
    outputTruthValue = val;
  }

}
