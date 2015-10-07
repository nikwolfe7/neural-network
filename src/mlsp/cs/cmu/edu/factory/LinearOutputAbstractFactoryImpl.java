package mlsp.cs.cmu.edu.factory;

import mlsp.cs.cmu.edu.elements.LinearOutput;
import mlsp.cs.cmu.edu.elements.Output;

public class LinearOutputAbstractFactoryImpl extends SigmoidNetworkAbstractFactoryImpl {

  @Override
  public Output getNewOutput() {
    return new LinearOutput();
  }

}
