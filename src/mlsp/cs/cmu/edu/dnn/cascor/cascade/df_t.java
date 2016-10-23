package mlsp.cs.cmu.edu.dnn.cascor.cascade;

import mlsp.cs.cmu.edu.dnn.cascor.parse.data_file_t;

/*  DF_T 
 * This structure is simply a container for a data file, so that the parsed
 * file can be linked into a linked list.                                   
 */
public class df_t {
	public data_file_t data;
	public df_t next; 
}
