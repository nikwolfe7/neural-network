package mlsp.cs.cmu.edu.structure;

public interface Layer {
	
	public void forward();

	public void backward();

	public double[] derivative();

	public double[] getOutput();

	public double[] getErrorTerm();

}
