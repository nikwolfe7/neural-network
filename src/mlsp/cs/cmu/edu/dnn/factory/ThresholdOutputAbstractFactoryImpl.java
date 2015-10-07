package mlsp.cs.cmu.edu.dnn.factory;

import mlsp.cs.cmu.edu.dnn.elements.Output;
import mlsp.cs.cmu.edu.dnn.elements.SimpleThresholdOutput;
import mlsp.cs.cmu.edu.dnn.elements.ThresholdOutput;

public class ThresholdOutputAbstractFactoryImpl extends SigmoidNetworkAbstractFactoryImpl {

  @Override
  public Output getNewOutput() {
    return new ThresholdOutput();
//    return new SimpleThresholdOutput();
  }

}
