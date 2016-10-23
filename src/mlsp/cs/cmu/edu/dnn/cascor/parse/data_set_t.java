package mlsp.cs.cmu.edu.dnn.cascor.parse;

/* DATA SET OBJECT - This structure contains (part of) the information needed to 
 * train a network to perform a specific task. The filename that the data came
 * from, the output types, number of inputs, outputs and the actual 
 * training vectors.  
 */
public class data_set_t {
	
	public String name;
	public int nPts;
	public double stdDev;
	public boolean predictOnly;
	public dv_t[] data;

}
