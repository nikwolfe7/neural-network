package mlsp.cs.cmu.edu.dnn.factory;

import mlsp.cs.cmu.edu.dnn.elements.Bias;
import mlsp.cs.cmu.edu.dnn.elements.Edge;
import mlsp.cs.cmu.edu.dnn.elements.Input;
import mlsp.cs.cmu.edu.dnn.elements.Neuron;
import mlsp.cs.cmu.edu.dnn.elements.Output;

public class SigmoidNetworkFactory implements NetworkElementAbstractFactory {

  @Override
  public Input getNewInput() {
    return new Input();
  }

  @Override
  public Output getNewOutput() {
    return new Output();
  }

  @Override
  public Neuron getNewNeuron() {
    return new Neuron();
  }

  @Override
  public Edge getNewEdge() {
	return new Edge(-5,5,0.1);
  }

  @Override
  public Bias getNewBias() {
    return new Bias();
  }

}
