package training;

import java.util.List;

public interface DataReader {
  
  List<DataInstance> getDataFromFile(String fileName, int inputDim, int outputDim);

}
