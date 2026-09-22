package number

// decimalUnits are the SI units, a thousand apart.
var decimalUnits = []string{"B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB", "RB", "QB"}

// binaryUnits are the IEC units, 1024 apart.
var binaryUnits = []string{"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB", "RiB", "QiB"}

// FileSize renders a byte count against the largest unit it reaches.
//
//	FileSize(1000, 0, -1, false) // "1 KB"
//	FileSize(1024, 0, -1, true)  // "1 KiB"
//	FileSize(1536, 2, -1, false) // "1.54 KB"
//
// All three arguments are required, because a boolean follows them, and a
// negative maxPrecision stands for the one that was not given.
//
// The unit steps up while the count is more than nine tenths of the base, which
// is why 999 bytes read "1 KB" and 900 of them read "900 B". A negative count
// never passes that test, so it keeps its unit: FileSize(-1000, 0, -1, false)
// is "-1,000 B".
func FileSize(bytes float64, precision, maxPrecision int, useBinaryPrefix bool) string {
	base, units := 1000.0, decimalUnits
	if useBinaryPrefix {
		base, units = 1024.0, binaryUnits
	}
	i := 0
	for bytes/base > 0.9 && i < len(units)-1 {
		bytes /= base
		i++
	}
	return formatWith(bytes, precision, maxPrecision) + " " + units[i]
}
