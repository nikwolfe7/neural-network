package mlsp.cs.cmu.edu.dnn.elements;

import mlsp.cs.cmu.edu.dnn.util.ActivationFunction;
import mlsp.cs.cmu.edu.dnn.util.LayerElementUtils;

public class SecondDerivativeOutput extends Output implements SecondDerivativeNetworkElement {

  private static final long serialVersionUID = -887267539965711250L;

  private double secondGradient;

  public SecondDerivativeOutput(Output output) {
    LayerElementUtils.convertOutput(output, this);
  }

  @Override
  public void backward() {
    super.backward();
    setSecondGradient(secondDerivative());
  }

  private void setSecondGradient(double val) {
    secondGradient = val;
  }

  @Override
  public double secondDerivative() {
    return ActivationFunction.sigmoidSecondDerivative(getOutput());
  }

  @Override
  public double getSecondGradient() {
    return secondGradient;
  }

}
