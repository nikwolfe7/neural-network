package mlsp.cs.cmu.edu.factory;

import mlsp.cs.cmu.edu.elements.Bias;
import mlsp.cs.cmu.edu.elements.Edge;
import mlsp.cs.cmu.edu.elements.Input;
import mlsp.cs.cmu.edu.elements.Neuron;
import mlsp.cs.cmu.edu.elements.Output;

public interface NetworkElementAbstractFactory {
	
	public Input getInputElement();
	
	public Edge getEdgeElement();
	
	public Neuron getNeuronElement();
	
	public Output getOutputElement();
	
	public Bias getBiasElement();

}
