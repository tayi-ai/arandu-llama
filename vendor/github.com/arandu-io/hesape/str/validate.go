package str

import (
	"encoding/json"
	"net"
	"regexp"
	"strings"
	"sync"
)

// IsJSON reports whether the string parses as JSON.
func IsJSON(value string) bool {
	return json.Valid([]byte(value))
}

// IsASCII reports whether every byte of the string is below 128.
func IsASCII(value string) bool { return isASCII(value) }

// IsURL reports whether the string is an absolute URL whose scheme is one of
// the protocols given, or one of the three hundred registered ones when none
// are given.
//
// The shape it accepts is a scheme, optional basic auth, a host that is a
// domain of at least two characters or an IPv4 address or a bracketed IPv6
// address, an optional port, and then path, query and fragment.
func IsURL(value string, protocols ...string) bool {
	if !urlPattern(protocols).MatchString(value) {
		return false
	}
	// The bracketed alternative is checked loosely by the pattern and exactly
	// here, because an IPv6 address written as a regular expression is a page
	// of parentheses that nobody can read or correct.
	if open := strings.IndexByte(value, '['); open >= 0 {
		closing := strings.IndexByte(value[open:], ']')
		if closing < 0 || net.ParseIP(value[open+1:open+closing]) == nil {
			return false
		}
	}
	return true
}

var (
	urlPatternOnce  sync.Once
	urlPatternCache *regexp.Regexp
	urlCustomMu     sync.Mutex
	urlCustomCache  = map[string]*regexp.Regexp{}
)

// urlPattern builds the expression for a protocol list, keeping the default one
// compiled once because it is three hundred alternatives long.
func urlPattern(protocols []string) *regexp.Regexp {
	if len(protocols) == 0 {
		urlPatternOnce.Do(func() { urlPatternCache = compileURLPattern(urlProtocols) })
		return urlPatternCache
	}
	list := strings.Join(protocols, "|")
	urlCustomMu.Lock()
	defer urlCustomMu.Unlock()
	if re, ok := urlCustomCache[list]; ok {
		return re
	}
	re := compileURLPattern(list)
	urlCustomCache[list] = re
	return re
}

func compileURLPattern(protocolList string) *regexp.Regexp {
	quoted := make([]string, 0, 8)
	for _, p := range strings.Split(protocolList, "|") {
		quoted = append(quoted, regexp.QuoteMeta(p))
	}
	const (
		userinfo = `(?:[_.\p{L}\p{N}-]|%[0-9A-Fa-f]{2})+`
		domain   = `[\p{L}\p{N}\p{S}\-_.]+(?:\.?(?:[\p{L}\p{N}]|xn--[\p{L}\p{N}-]+)+\.?)`
		ipv4     = `\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}`
		ipv6     = `\[[0-9A-Fa-f:.]{2,45}\]`
		path     = `(?:/(?:[\p{L}\p{N}\-._~!$&'()*+,;=:@]|%[0-9A-Fa-f]{2})*)*`
		query    = `(?:\?(?:[\p{L}\p{N}\-._~!$&'\[\]()*+,;=:@/?]|%[0-9A-Fa-f]{2})*)?`
		fragment = `(?:#(?:[\p{L}\p{N}\-._~!$&'()*+,;=:@/?]|%[0-9A-Fa-f]{2})*)?`
	)
	return regexp.MustCompile(`(?i)^(?:` + strings.Join(quoted, "|") + `)://` +
		`(?:(?:` + userinfo + `:)?` + userinfo + `@)?` +
		`(?:` + domain + `|` + ipv4 + `|` + ipv6 + `)` +
		`(?::[0-9]+)?` + path + query + fragment + `\z`)
}

// urlProtocols is the default protocol list IsURL carries: the IANA registry of
// URI schemes, pipe separated.
const urlProtocols = "aaa|aaas|about|acap|acct|acd|acr|adiumxtra|adt|afp|afs|aim|amss|android|appdata|apt|ark|attachment|aw|barion|beshare|bitcoin|bitcoincash|blob|bolo|browserext|calculator|callto|cap|cast|casts|chrome|chrome-extension|cid|coap|coap+tcp|coap+ws|coaps|coaps+tcp|coaps+ws|com-eventbrite-attendee|content|conti|crid|cvs|dab|data|dav|diaspora|dict|did|dis|dlna-playcontainer|dlna-playsingle|dns|dntp|dpp|drm|drop|dtn|dvb|ed2k|elsi|example|facetime|fax|feed|feedready|file|filesystem|finger|first-run-pen-experience|fish|fm|ftp|fuchsia-pkg|geo|gg|git|gizmoproject|go|gopher|graph|gtalk|h323|ham|hcap|hcp|http|https|hxxp|hxxps|hydrazone|iax|icap|icon|im|imap|info|iotdisco|ipn|ipp|ipps|irc|irc6|ircs|iris|iris.beep|iris.lwz|iris.xpc|iris.xpcs|isostore|itms|jabber|jar|jms|keyparc|lastfm|ldap|ldaps|leaptofrogans|lorawan|lvlt|magnet|mailserver|mailto|maps|market|message|mid|mms|modem|mongodb|moz|ms-access|ms-browser-extension|ms-calculator|ms-drive-to|ms-enrollment|ms-excel|ms-eyecontrolspeech|ms-gamebarservices|ms-gamingoverlay|ms-getoffice|ms-help|ms-infopath|ms-inputapp|ms-lockscreencomponent-config|ms-media-stream-id|ms-mixedrealitycapture|ms-mobileplans|ms-officeapp|ms-people|ms-project|ms-powerpoint|ms-publisher|ms-restoretabcompanion|ms-screenclip|ms-screensketch|ms-search|ms-search-repair|ms-secondary-screen-controller|ms-secondary-screen-setup|ms-settings|ms-settings-airplanemode|ms-settings-bluetooth|ms-settings-camera|ms-settings-cellular|ms-settings-cloudstorage|ms-settings-connectabledevices|ms-settings-displays-topology|ms-settings-emailandaccounts|ms-settings-language|ms-settings-location|ms-settings-lock|ms-settings-nfctransactions|ms-settings-notifications|ms-settings-power|ms-settings-privacy|ms-settings-proximity|ms-settings-screenrotation|ms-settings-wifi|ms-settings-workplace|ms-spd|ms-sttoverlay|ms-transit-to|ms-useractivityset|ms-virtualtouchpad|ms-visio|ms-walk-to|ms-whiteboard|ms-whiteboard-cmd|ms-word|msnim|msrp|msrps|mss|mtqp|mumble|mupdate|mvn|news|nfs|ni|nih|nntp|notes|ocf|oid|onenote|onenote-cmd|opaquelocktoken|openpgp4fpr|pack|palm|paparazzi|payto|pkcs11|platform|pop|pres|prospero|proxy|pwid|psyc|pttp|qb|query|redis|rediss|reload|res|resource|rmi|rsync|rtmfp|rtmp|rtsp|rtsps|rtspu|s3|secondlife|service|session|sftp|sgn|shttp|sieve|simpleledger|sip|sips|skype|smb|sms|smtp|snews|snmp|soap.beep|soap.beeps|soldat|spiffe|spotify|ssh|steam|stun|stuns|submit|svn|tag|teamspeak|tel|teliaeid|telnet|tftp|tg|things|thismessage|tip|tn3270|tool|ts3server|turn|turns|tv|udp|unreal|urn|ut2004|v-event|vemmi|ventrilo|videotex|vnc|view-source|wais|webcal|wpid|ws|wss|wtai|wyciwyg|xcon|xcon-userid|xfire|xmlrpc.beep|xmlrpc.beeps|xmpp|xri|ymsgr|z39.50|z39.50r|z39.50s"
