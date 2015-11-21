package cs224n.deep;
import java.util.*;

public class Datum {

	public static final String[] POSSIBLE_LABELS = {"O", "LOC", "MISC", "ORG", "PER"};
	public static final String DEFAULT_LABEL = "O";
	public static final String SEPARATE_WORD = "$S$";
	public static final String START_WORD = "<s>";
	public static final String END_WORD = "</s>";
	public final String word;
	public final String label;

	public Datum(String word, String label) {
		this.word = word;
		this.label = label;
	}
	
	@Override
	public String toString()
	{
		return this.word + ": " + this.label;
	}
}