package ner;

import java.util.Properties;

import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.sequences.SeqClassifierFlags;
import edu.stanford.nlp.util.StringUtils;

public class RunTrainer {
	
	public static void main(String[] args) throws Exception {

	   String modelOutput="../NERProject/output/outputText.txt";
	   String prop="../NERProject/src/ner/properties.txt";
	   String trainingFilePath="../NERProject/data/namedEntity.txt";
	   RunTrainer rt = new RunTrainer();
	   rt.trainAndWrite(modelOutput,prop,trainingFilePath);
	   
	   String[] tests = new String[] {"watch wrist", "comb hair","band wrist", "barrel clothes", "roman earing"};
	   CRFClassifier model=rt.getModel(modelOutput);
	   
	   for (String item : tests) {
	     rt.doTagging(model, item);
	   }
			   
	 }
	
	public void trainAndWrite(String modelOutPath, String prop, String trainingFilepath) {
		   Properties props = StringUtils.propFileToProperties(prop);
		   props.setProperty("serializeTo", modelOutPath);
		   //if input use that, else use from properties file.
		   if (trainingFilepath != null) {
		       props.setProperty("trainFile", trainingFilepath);
		   }
		   SeqClassifierFlags flags = new SeqClassifierFlags(props);
		   CRFClassifier<CoreLabel> crf = new CRFClassifier<>(flags);
		   crf.train();
		   crf.serializeClassifier(modelOutPath);
		}
	
	public CRFClassifier getModel(String modelPath) {
	    return CRFClassifier.getClassifierNoExceptions(modelPath);
	}
	
	public void doTagging(CRFClassifier model, String input) {
		  input = input.trim();
		  System.out.println(input + "=>"  +  model.classifyToString(input));
		}


}
