package mlsp.cs.cmu.edu.dnn.elements;

import mlsp.cs.cmu.edu.dnn.util.ActivationFunction;
import mlsp.cs.cmu.edu.dnn.util.CostFunction;

public class CrossEntropySoftmaxOutput extends Output {

  private static final long serialVersionUID = -6410216992964367341L;
  
  @Override
  public void backward() {
    double gradient = CostFunction.crossEntropyDerivative(getOutput(), getTruthValue());
    setGradient(gradient);
  }
  
  @Override
  public void forward() {
    double sum = 0;
    for(NetworkElement e : getIncomingElements())
      sum += e.getOutput();
    setOutput(ActivationFunction.exp(sum));
  }
  
  public void softMax(double divisor) {
    setOutput(getOutput() * divisor);
  }

}
