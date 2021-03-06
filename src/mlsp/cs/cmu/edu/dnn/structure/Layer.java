package mlsp.cs.cmu.edu.dnn.structure;

import java.io.Serializable;

import mlsp.cs.cmu.edu.dnn.elements.NetworkElement;

public interface Layer extends Serializable {
	
	public void forward();

	public void backward();

	public double[] derivative();

	public double[] getOutput();

	public double[] getGradient();
	
	public NetworkElement[] getElements();
	
	public void addNetworkElements(NetworkElement... elements);
	
	public void removeNetworkElement(NetworkElement e);
	
	public int size();

}
