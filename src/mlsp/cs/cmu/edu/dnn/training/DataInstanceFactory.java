package mlsp.cs.cmu.edu.dnn.training;

import java.util.List;

public interface DataInstanceFactory {
	
	public List<DataInstance> getTrainingInstances();
	
	public List<DataInstance> getTestingInstances();

}
