package mlsp.cs.cmu.edu.factory;

import mlsp.cs.cmu.edu.elements.Output;
import mlsp.cs.cmu.edu.elements.SimpleThresholdOutput;
import mlsp.cs.cmu.edu.elements.ThresholdOutput;

public class ThresholdOutputAbstractFactoryImpl extends SigmoidNetworkAbstractFactoryImpl {

  @Override
  public Output getNewOutput() {
    return new ThresholdOutput();
//    return new SimpleThresholdOutput();
  }

}
