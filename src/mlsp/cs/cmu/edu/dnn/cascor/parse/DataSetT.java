package mlsp.cs.cmu.edu.dnn.cascor.parse;

/* DATA SET OBJECT - This structure contains (part of) the information needed to 
 * train a network to perform a specific task. The filename that the data came
 * from, the output types, number of inputs, outputs and the actual 
 * training vectors.  
 */
public class DataSetT {
	
	public String name;
	public int nPts;
	public double stdDev;
	public boolean predictOnly;
	public DataValueT[] data;

}