package mlsp.cs.cmu.edu.dnn.factory;

import mlsp.cs.cmu.edu.dnn.elements.LinearOutput;
import mlsp.cs.cmu.edu.dnn.elements.Output;

public class LinearOutputElementFactory extends SigmoidNetworkElementFactory {

  @Override
  public Output getNewOutput() {
    return new LinearOutput();
  }

}
