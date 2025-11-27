package stem

import "testing"

func TestStem(t *testing.T) {
	testCases := []struct {
		word     string
		expected string
	}{
		{"running", "runn"},
		{"jumps", "jump"},
		{"studying", "study"},
		{"boxes", "box"},
		{"happily", "happi"},
		{"", ""},
	}

	for _, tc := range testCases {
		t.Run(tc.word, func(t *testing.T) {
			actual := Stem(tc.word)
			if actual != tc.expected {
				t.Errorf("Stem(%q) = %q; expected %q", tc.word, actual, tc.expected)
			}
		})
	}
}
