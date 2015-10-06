package mlsp.cs.cmu.edu.structure;

import mlsp.cs.cmu.edu.elements.NetworkElement;

public interface Layer {
	
	public void forward();

	public void backward();

	public double[] derivative();

	public double[] getOutput();

	public double[] getErrorTerm();
	
	public NetworkElement[] getElements();
	
	public int size();

}
