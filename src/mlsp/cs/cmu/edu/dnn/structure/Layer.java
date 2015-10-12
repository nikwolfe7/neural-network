package mlsp.cs.cmu.edu.dnn.structure;

import java.io.Serializable;

import mlsp.cs.cmu.edu.dnn.elements.NetworkElement;

public interface Layer extends Serializable {
	
	public void forward();

	public void backward();

	public double[] derivative();

	public double[] getOutput();

	public double[] getErrorTerm();
	
	public NetworkElement[] getElements();
	
	public void addNetworkElements(NetworkElement... elements);
	
	public int size();

}
