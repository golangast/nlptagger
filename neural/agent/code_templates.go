package agent

import (
	"fmt"
	"strings"
)

// CodeTemplate represents a code generation template
type CodeTemplate struct {
	Name        string
	Description string
	Files       map[string]string // filename -> content template
}

// TemplateRegistry manages code templates
type TemplateRegistry struct {
	templates map[string]*CodeTemplate
}

// NewTemplateRegistry creates a new template registry
func NewTemplateRegistry() *TemplateRegistry {
	registry := &TemplateRegistry{
		templates: make(map[string]*CodeTemplate),
	}
	registry.registerBuiltinTemplates()
	return registry
}

// registerBuiltinTemplates adds built-in code templates
func (tr *TemplateRegistry) registerBuiltinTemplates() {
	// Webserver template
	tr.templates["webserver"] = &CodeTemplate{
		Name:        "webserver",
		Description: "Simple Go web server",
		Files: map[string]string{
			"main.go": `package main

import (
	"fmt"
	"log"
	"net/http"
)

func main() {
	http.HandleFunc("/", homeHandler)
	
	port := "8080"
	fmt.Printf("Server starting on http://localhost:%s\n", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func homeHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello, World! Server is running.")
}
`,
			"go.mod": `module {{.ProjectName}}

go {{.GoVersion}}
`,
		},
	}

	// REST API template
	tr.templates["rest_api"] = &CodeTemplate{
		Name:        "rest_api",
		Description: "REST API with basic CRUD operations",
		Files: map[string]string{
			"main.go": `package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
)

type Item struct {
	ID   string ` + "`json:\"id\"`" + `
	Name string ` + "`json:\"name\"`" + `
}

var (
	items = make(map[string]Item)
	mu    sync.RWMutex
)

func main() {
	http.HandleFunc("/items", itemsHandler)
	http.HandleFunc("/items/", itemHandler)
	http.HandleFunc("/health", healthHandler)
	
	port := "8080"
	fmt.Printf("REST API server starting on http://localhost:%s\n", port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
}

func itemsHandler(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		mu.RLock()
		defer mu.RUnlock()
		
		itemList := make([]Item, 0, len(items))
		for _, item := range items {
			itemList = append(itemList, item)
		}
		json.NewEncoder(w).Encode(itemList)
		
	case http.MethodPost:
		var item Item
		if err := json.NewDecoder(r.Body).Decode(&item); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		
		mu.Lock()
		items[item.ID] = item
		mu.Unlock()
		
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(item)
		
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func itemHandler(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.Path, "/items/")
	
	switch r.Method {
	case http.MethodGet:
		mu.RLock()
		item, exists := items[id]
		mu.RUnlock()
		
		if !exists {
			http.Error(w, "Item not found", http.StatusNotFound)
			return
		}
		json.NewEncoder(w).Encode(item)
		
	case http.MethodDelete:
		mu.Lock()
		delete(items, id)
		mu.Unlock()
		
		w.WriteHeader(http.StatusNoContent)
		
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}
`,
			"go.mod": `module {{.ProjectName}}

go {{.GoVersion}}
`,
			"README.md": `# {{.ProjectName}}

A static HTML website.

## Files
- index.html
- style.css
- script.js

## Running
Open index.html in your browser.
`,
		},
	}

	// Landing page template
	tr.templates["landing_page"] = &CodeTemplate{
		Name:        "landing_page",
		Description: "Modern landing page with hero section",
		Files: map[string]string{
			"index.html": `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{.ProjectName}} - Landing Page</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <nav class="navbar">
        <div class="nav-container">
            <div class="logo">{{.ProjectName}}</div>
            <ul class="nav-menu">
                <li><a href="#features">Features</a></li>
                <li><a href="#about">About</a></li>
                <li><a href="#contact" class="cta-button">Get Started</a></li>
            </ul>
        </div>
    </nav>

    <section class="hero">
        <div class="hero-content">
            <h1>Welcome to {{.ProjectName}}</h1>
            <p class="hero-subtitle">The best solution for your needs</p>
            <button class="cta-button large">Get Started Now</button>
        </div>
    </section>

    <section id="features" class="features">
        <h2>Amazing Features</h2>
        <div class="feature-grid">
            <div class="feature">
                <div class="feature-icon">🚀</div>
                <h3>Fast & Reliable</h3>
                <p>Lightning-fast performance you can count on</p>
            </div>
            <div class="feature">
                <div class="feature-icon">🎨</div>
                <h3>Beautiful Design</h3>
                <p>Modern, clean interface that users love</p>
            </div>
            <div class="feature">
                <div class="feature-icon">🔒</div>
                <h3>Secure</h3>
                <p>Your data is safe with enterprise-grade security</p>
            </div>
        </div>
    </section>

    <footer>
        <p>&copy; 2025 {{.ProjectName}}. All rights reserved.</p>
    </footer>

    <script src="script.js"></script>
</body>
</html>
`,
			"style.css": `* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #333;
    overflow-x: hidden;
}

.navbar {
    background: white;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    position: fixed;
    width: 100%;
    top: 0;
    z-index: 1000;
}

.nav-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    font-size: 1.5rem;
    font-weight: bold;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.nav-menu {
    display: flex;
    list-style: none;
    gap: 2rem;
    align-items: center;
}

.nav-menu a {
    text-decoration: none;
    color: #333;
    transition: color 0.3s;
}

.nav-menu a:hover {
    color: #667eea;
}

.hero {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    text-align: center;
    padding: 2rem;
}

.hero-content h1 {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    animation: fadeInUp 1s;
}

.hero-subtitle {
    font-size: 1.5rem;
    margin-bottom: 2rem;
    opacity: 0.9;
    animation: fadeInUp 1s 0.2s both;
}

.cta-button {
    background: white;
    color: #667eea;
    padding: 1rem 2rem;
    border: none;
    border-radius: 50px;
    font-size: 1rem;
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s;
    text-decoration: none;
    display: inline-block;
}

.cta-button.large {
    padding: 1.25rem 3rem;
    font-size: 1.1rem;
}

.cta-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}

.features {
    padding: 5rem 2rem;
    max-width: 1200px;
    margin: 0 auto;
}

.features h2 {
    text-align: center;
    font-size: 2.5rem;
    margin-bottom: 3rem;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 3rem;
}

.feature {
    text-align: center;
    padding: 2rem;
    border-radius: 10px;
    transition: transform 0.3s;
}

.feature:hover {
    transform: translateY(-10px);
}

.feature-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
}

.feature h3 {
    font-size: 1.5rem;
    margin-bottom: 1rem;
    color: #667eea;
}

footer {
    background: #333;
    color: white;
    text-align: center;
    padding: 2rem;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
`,
			"script.js": `console.log('{{.ProjectName}} loaded!');

// Smooth scrolling
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});

// CTA button interaction
document.querySelectorAll('.cta-button').forEach(button => {
    button.addEventListener('click', () => {
        alert('Thanks for your interest in {{.ProjectName}}!');
    });
});
`,
			"README.md": `# {{.ProjectName}}

A static HTML website.

## Files
- index.html
- style.css  
- script.js

## Running
Open index.html in your browser.
`,
		},
	}

	// Handlers template
	tr.templates["handlers"] = &CodeTemplate{
		Name:        "handlers",
		Description: "Generic HTTP handlers",
		Files: map[string]string{
			"handlers.go": `package main

import (
	"fmt"
	"net/http"
)

func helloHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello, handlers!")
}
`,
		},
	}

	// Go project template
	tr.templates["go_project"] = &CodeTemplate{
		Name:        "go_project",
		Description: "Empty Go project",
		Files:       map[string]string{},
	}

	// HTML template
	tr.templates["html"] = &CodeTemplate{
		Name:        "html",
		Description: "Simple HTML project with templates folder",
		Files: map[string]string{
			"templates/index.html": `<!DOCTYPE html>
<html>
<head>
	<title>{{.ProjectName}}</title>
</head>
<body>
	<h1>Hello, {{.ProjectName}}!</h1>
</body>
</html>`,
		},
	}
}

// GetTemplate retrieves a template by name
func (tr *TemplateRegistry) GetTemplate(name string) (*CodeTemplate, bool) {
	template, exists := tr.templates[name]
	return template, exists
}

// RenderTemplate renders a template with given parameters
func (tr *TemplateRegistry) RenderTemplate(templateName string, params map[string]interface{}) (map[string]string, error) {
	template, exists := tr.GetTemplate(templateName)
	if !exists {
		return nil, fmt.Errorf("template not found: %s", templateName)
	}

	// Set defaults
	if params["ProjectName"] == nil {
		params["ProjectName"] = "myproject"
	}
	if params["GoVersion"] == nil {
		params["GoVersion"] = "1.21"
	}

	// Render each file
	rendered := make(map[string]string)
	for filename, content := range template.Files {
		rendered[filename] = renderString(content, params)
	}

	return rendered, nil
}

// renderString performs simple string replacement for template variables
func renderString(template string, params map[string]interface{}) string {
	result := template
	for key, value := range params {
		placeholder := fmt.Sprintf("{{.%s}}", key)
		result = strings.ReplaceAll(result, placeholder, fmt.Sprintf("%v", value))
	}
	return result
}

// InferTemplate infers which template to use based on goal description
func (tr *TemplateRegistry) InferTemplate(description string) string {
	lower := strings.ToLower(description)

	// HTML/Frontend templates
	if strings.Contains(lower, "landing page") || strings.Contains(lower, "landing") {
		return "landing_page"
	}
	if strings.Contains(lower, "html") || strings.Contains(lower, "website") || strings.Contains(lower, "web page") {
		return "html"
	}

	// Backend templates
	if strings.Contains(lower, "rest") || strings.Contains(lower, "api") {
		return "rest_api"
	}
	if strings.Contains(lower, "webserver") || strings.Contains(lower, "web server") || strings.Contains(lower, "http server") {
		return "webserver"
	}
	if strings.Contains(lower, "handler") {
		return "handlers"
	}

	// Default to generic Go project
	return "go_project"
}
