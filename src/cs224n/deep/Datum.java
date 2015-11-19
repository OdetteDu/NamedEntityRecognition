package cs224n.deep;
import java.util.*;

public class Datum {

	public static final String[] POSSIBLE_LABELS = {"O", "LOC", "MISC", "ORG", "PER"};
	public final String word;
	public final String label;

	public Datum(String word, String label) {
		this.word = word;
		this.label = label;
	}
}