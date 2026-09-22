package routing

// --- Route fields added by the URL and bindings code ---

// action is the controller action string ("Controller@method").
// It is set by the controller-based route registration and read by
// UrlGenerator.Action.
var (
	routeActionName   = make(map[*Route]string)
	routeController   = make(map[*Route]string)
	routeActionMethod = make(map[*Route]string)
)

// SetActionName stores the action name for URL generation via action().
func (r *Route) SetActionName(action string) {
	routeActionName[r] = action
}

// SetControllerMethod stores controller and method for this route.
func (r *Route) SetControllerMethod(controller, method string) {
	routeController[r] = controller
	routeActionMethod[r] = method
}

// actionName returns the controller action string for this route.
func (r *Route) actionName() string {
	if a, ok := routeActionName[r]; ok {
		return a
	}
	ctrl := routeController[r]
	m := routeActionMethod[r]
	if ctrl != "" {
		return ctrl + "@" + m
	}
	return ""
}
