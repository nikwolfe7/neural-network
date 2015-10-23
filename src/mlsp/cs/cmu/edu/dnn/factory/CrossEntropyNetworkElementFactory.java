package mlsp.cs.cmu.edu.dnn.factory;

import mlsp.cs.cmu.edu.dnn.elements.CrossEntropyOutput;
import mlsp.cs.cmu.edu.dnn.elements.Output;


public class CrossEntropyNetworkElementFactory extends SigmoidNetworkElementFactory {

  @Override
  public Output getNewOutput() {
    return new CrossEntropyOutput();
  }
  
}
