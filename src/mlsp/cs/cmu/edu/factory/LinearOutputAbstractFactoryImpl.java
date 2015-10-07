package mlsp.cs.cmu.edu.factory;

import mlsp.cs.cmu.edu.elements.Bias;
import mlsp.cs.cmu.edu.elements.Edge;
import mlsp.cs.cmu.edu.elements.Input;
import mlsp.cs.cmu.edu.elements.LinearOutput;
import mlsp.cs.cmu.edu.elements.Neuron;
import mlsp.cs.cmu.edu.elements.Output;

public class LinearOutputAbstractFactoryImpl implements NetworkElementAbstractFactory {

	@Override
	  public Input getNewInput() {
	    return new Input();
	  }

	  @Override
	  public Output getNewOutput() {
	    return new LinearOutput();
	  }

	  @Override
	  public Neuron getNewNeuron() {
	    return new Neuron();
	  }

	  @Override
	  public Edge getNewEdge() {
	    return new Edge();
	  }

	  @Override
	  public Bias getNewBias() {
	    return new Bias();
	  }

}
