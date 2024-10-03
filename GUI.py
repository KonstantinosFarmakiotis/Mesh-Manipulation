import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from open3d.visualization.gui import MouseEvent, KeyEvent
from open3d.visualization.rendering import Camera
import sys
import os
import numpy as np
import utility as U
import platform
from scipy.sparse.linalg import eigs, eigsh
from scipy.linalg import eigh
import time as t


isMacOS = (platform.system() == "Darwin")

class AppWindow:
    
    MENU_OPEN = 1
    MENU_QUIT = 2

    def __init__(self, width, height):

        resource_path = gui.Application.instance.resource_path
        self.w_width = width
        self.w_height = height
        self.first_click = True

        #boilerplate - initialize window & scene
        self.window = gui.Application.instance.create_window("Test", width, height)
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
                
        #initializing menubar
        self._init_menubar()

        #basic layout
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)

        #set mouse and key callbacks
        self._scene.set_on_key(self._on_key_pressed)
        self.hook = False #hook for keys so the function doesnt run 2 times per press
        self._scene.set_on_mouse(self._on_mouse_pressed)

        #geometry container for future reference
        self.geometry = None
        self.vertices = None
        self.triangles = None
        self.tree = None
        self.selected_vertex = None
        self.eigenvectors = None
        self.which = "mesh"
        self.current_eigenvector = 0

        #materials
        self.matlit = rendering.MaterialRecord()
        self.matlit.shader = "defaultLit"
        self.matunlit = rendering.MaterialRecord()
        self.matunlit.shader = "defaultUnlit"
        self.matline = rendering.MaterialRecord()
        self.matline.shader = "unlitLine"
        self.matline.base_color = np.array([0, 0, 0, 1], dtype=np.float32)
        self.matline.line_width = 2

        #creation of skeletonization variables
        self.WL = 0
        self.WH = 0
        self.iterator = 0
        self.vertex_areas = 0
        
        #creation of coating transfer variables
        self.turn = 0
        self.coat = np.array([])
        self.first_vector = np.array([])
        
    def _init_menubar(self):

        if gui.Application.instance.menubar is None:

            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("Quit", AppWindow.MENU_QUIT)

            file_menu = gui.Menu()
            file_menu.add_item("Open...", AppWindow.MENU_OPEN)

            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", AppWindow.MENU_QUIT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)

            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        self.window.set_on_menu_item_activated(AppWindow.MENU_OPEN, self._on_menu_open)
        self.window.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)

    def _on_menu_open(self):
        
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)
    
    def _on_file_dialog_cancel(self):
        self.window.close_dialog()
    
    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _preprocess(self, m):

        vertices, triangles = np.asarray(m.vertices), np.asarray(m.triangles)
     
        #centering
        vertices = vertices - vertices.mean(0)
        
        #unit_sphere_normalization
        norm = np.max((vertices * vertices).sum(-1))
        vertices = vertices / np.sqrt(norm)

        return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))

    def _find_match(self, query):

        if self.geometry is not None:

            ind = 0
            if self.tree is not None:
                _, ind, _ = self.tree.search_knn_vector_3d(query, 1)
                ind = int(np.asarray(ind)[0])
                self.selected_vertex = ind
                return self.vertices[ind]
                
            else:
                d = self.vertices - query
                d = np.argmin((d * d).sum(-1))
                self.selected_vertex = d
                return self.vertices[ind]

    def load(self, path):

        #clearing scene
        self._scene.scene.clear_geometry()

        #setting iterator to zero to show how many times we have performed skeletonization to the mesh
        self.iterator = 0
        
        #reading geometry type
        geometry_type = o3d.io.read_file_geometry_type(path)

        #checking the type of geometry
        if geometry_type & o3d.io.CONTAINS_TRIANGLES:
            self.geometry = o3d.io.read_triangle_model(path).meshes[0].mesh

            
        if self.geometry is None:
            print("[Info]", path, "appears to not be a triangle mesh")
            return 
        else:
            #preprocessing and setting geometry
            self.geometry = self._preprocess(self.geometry)
            self.line_geometry = o3d.geometry.LineSet.create_from_triangle_mesh(self.geometry)

            #setting vertex and triangle data for easy access
            self.vertices = np.asarray(self.geometry.vertices)
            self.triangles = np.asarray(self.geometry.triangles)

            #initializing kd-tree for quick searches
            self.tree = o3d.geometry.KDTreeFlann(self.geometry)


            #adding mesh to the scene and reconfiguring camera
            self.geometry.compute_vertex_normals()
            vertex_normals = np.asarray(self.geometry.vertex_normals)
            vertex_colors = (vertex_normals+0.5) / 3
            # vertex_colors = vertex_normals @ np.array([0.2989, 0.5870, 0.1140])
            # vertex_colors = np.repeat(vertex_colors[:, np.newaxis], 3, axis=1)
            self.geometry.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            self._scene.scene.add_geometry("__model__", self.geometry, self.matunlit)
            bounds = self._scene.scene.bounding_box
            self._scene.setup_camera(60, bounds, bounds.get_center())


    def _on_layout(self, layout_context):
        
        r = self.window.content_rect
        self._scene.frame = r

    def _on_key_pressed(self, event):

        self.hook = not self.hook #hook for keys so the function doesnt run 2 times per press
        if self.hook : 
            return gui.Widget.EventCallbackResult.IGNORED
        
        print("key pressed: ", event.key)
        
        #T key - toggle mesh or lineset
        if event.key == 116 :
            self.which = "line" if self.which == "mesh" else "mesh"
            print("mode = ", self.which)
            self._redraw_scene()
            return gui.Widget.EventCallbackResult.HANDLED

        #A key - smooth universal
        elif event.key == 97:
            for i in range(5):
                self._taubin_smoothing()
            print("taubin smoothing.\n")
            return gui.Widget.EventCallbackResult.HANDLED
        
        #B key - smooth local
        elif event.key == 98:
            for i in range(5):
                self._taubin_smoothing_local()
            print("taubin smoothing local.\n")
            return gui.Widget.EventCallbackResult.HANDLED

        #C key - skeletonization universal
        elif event.key == 99:
            self._skeletonization()
            print("skeletonization universal\n")
            return gui.Widget.EventCallbackResult.HANDLED
        
        #D key - skeletonization local
        elif event.key == 100:
            self._skeletonization_local()
            print("skeletonization local\n")
            return gui.Widget.EventCallbackResult.HANDLED

        #E key - pointcloud skeletonization
        elif event.key == 101:
            self._pc_skeletonization()
            print("pointcloud skeletonization\n")
            return gui.Widget.EventCallbackResult.HANDLED
        
        #F key - edge decrease
        elif event.key == 102:
            print("\nedge decreasing\n")
            for i in range(1):
                self._edge_decrease()
            return gui.Widget.EventCallbackResult.HANDLED
        
        #G key - edge increase
        elif event.key == 103:
            print("\nedge increasing\n")
            for i in range(1):
                self._edge_increase()
            return gui.Widget.EventCallbackResult.HANDLED
        
        #H key - coating transfer
        elif event.key == 104:
            print("\ncoating transfer\n")
            self._coating_transfer()
            return gui.Widget.EventCallbackResult.HANDLED
        
        #number key -> ring neighborhood
        # if event.key <= 57 and event.key >= 48:
        #     self._show_k_ring(event.key-48)

        #C key - delta coordinates
        # if event.key == 99:
        #     self._show_delta_coordinates()
        #     return gui.Widget.EventCallbackResult.HANDLED

        #S key - eigendecomposition
        # elif event.key == 115:
        #     self._show_eigendecomposition()
        #     return gui.Widget.EventCallbackResult.HANDLED
        
        #V key - reset geometry and redraw scene
        # if event.key == 118:
        #     self._reset_geometry()
        #     self._redraw_scene()
        #     return gui.Widget.EventCallbackResult.HANDLED
        
        #R key - eigenvector visualization mode
        # elif event.key == 114:
        #     self._calc_eigenvectors()
        #     print("eigenvectors calculated.")
        #     return gui.Widget.EventCallbackResult.HANDLED
        
        #L key - laplace mode
        # if event.key == 108:
        #     for i in range(50):
        #         self._laplace_smoothing()
        #     print("laplace smoothing.")
        #     return gui.Widget.EventCallbackResult.HANDLED
        
        #left or bottom arrow keys - decrease eigenvector counter
        # elif event.key == 263 or event.key == 266:
        #     self.current_eigenvector = self.current_eigenvector -1 if self.current_eigenvector > 0 else 0
        #     print("current eigenvector: ", self.current_eigenvector)
        #     return gui.Widget.EventCallbackResult.HANDLED
        
        #right or up arrow keys - increase eigenvector counter
        # elif event.key == 264 or event.key == 265:
        #     self.current_eigenvector = self.current_eigenvector +1 if self.current_eigenvector < self.vertices.shape[0]-1 else self.vertices.shape[0]-1
        #     print("current eigenvector: ", self.current_eigenvector)
        #     return gui.Widget.EventCallbackResult.HANDLED

        #enter key - show eigenvector
        # elif event.key == 10:
        #     self._show_eigenvector()
        #     return gui.Widget.EventCallbackResult.HANDLED

        else:
            return gui.Widget.EventCallbackResult.IGNORED

    def _on_mouse_pressed(self, event):

        # interfere with manipulating the scene.
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.CTRL):

            def depth_callback(depth_image):
                # Coordinates are expressed in absolute coordinates of the
                # window, but to dereference the image correctly we need them
                # relative to the origin of the widget. Note that even if the
                # scene widget is the only thing in the window, if a menubar
                # exists it also takes up space in the window (except on macOS).
                x = event.x - self._scene.frame.x
                y = event.y - self._scene.frame.y
                # Note that np.asarray() reverses the axes.
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:  # clicked on nothing (i.e. the far plane)
                    self.selected_vertex = None
                else:
                    #finding point on the mesh
                    world = self._scene.scene.camera.unproject(
                        x, y, depth, self._scene.frame.width,
                        self._scene.frame.height)
                    
                    #finding closest mesh vertex
                    match = self._find_match(world)

                    #adding a sphere
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01).paint_uniform_color(np.array([1,0,0])).translate(match)
                    self._scene.scene.remove_geometry("__point__")
                    self._scene.scene.add_geometry("__point__", sphere, self.matlit)


            self._scene.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
    
        elif event.type == MouseEvent.Type.BUTTON_DOWN:
            return gui.Widget.EventCallbackResult.HANDLED
        
        return gui.Widget.EventCallbackResult.HANDLED

    def _set_projection(self):
        self._scene.scene.camera.set_projection(
            Camera.Projection(1), -2.0, 2.0, -2.0, 2.0, 0.1, 100.0
        )

    def _reset_geometry(self):

        self.geometry = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self.vertices),
            o3d.utility.Vector3iVector(self.triangles)
        )

        self.line_geometry = o3d.geometry.LineSet.create_from_triangle_mesh(self.geometry)

    def _redraw_scene(self):

        #clearing scene
        self._scene.scene.clear_geometry()
        
        #if line mode then draw lineset
        if self.which == "line":
            self._scene.scene.add_geometry("__model__", self.line_geometry, self.matline)
        elif self.which == "mesh":
            self._scene.scene.add_geometry("__model__", self.geometry, self.matunlit)

    def _show_delta_coordinates(self):

        #calculate delta coordinates
        delta = U.delta_coordinates(self.vertices, self.triangles, use_laplacian=True)

        #calculating norm of delta vector
        norm = np.sqrt((delta * delta).sum(-1))

        #linear transformation
        norm = (norm - norm.min()) / (norm.max() - norm.min())

        #coloring the mesh
        colors = U.sample_colormap(norm)
        # colors = np.zeros_like(self.vertices)
        # colors[:,0] = norm

        self.geometry.vertex_colors = o3d.utility.Vector3dVector(colors)

        self._redraw_scene()

    def _show_k_ring(self, k):

        if self.selected_vertex is not None:
            print(f"finding {k}-ring neighbors")
            neighbors = U.k_ring_adjacency(self.selected_vertex, self.triangles, k)
            # neighbors = U.k_ring_recursive(self.selected_vertex, self.triangles, k)

            colors = np.zeros_like(self.vertices)
            colors[neighbors, 0] = 1

            self.geometry.vertex_colors = o3d.utility.Vector3dVector(colors)

            self._redraw_scene()

    def _show_eigendecomposition(self):

        if self.geometry is not None:
            #constants
            num_components = self.vertices.shape[0]
            keep_percent = 0.1
            keep_components = int(num_components * keep_percent)
            discard_components = int(num_components * (1-keep_percent))

            #calculating the graph laplacian
            L = U.graph_laplacian(self.triangles).astype(np.float64)

            #eigen decomposition of symmetric matrix -> SM means return the smallest eigenvalues
            vals, vecs = eigsh(L, k=keep_components, which='SM')

            #forming the eigenvector matrix with only the significant components
            U_k = vecs#[:, 0:keep_components]
            V_filtered = U_k @ (U_k.T @ self.vertices)

            #setting the vertices to be the filtered ones
            self.geometry.vertices = o3d.utility.Vector3dVector(V_filtered)

            #redrawing to see the difference
            self._redraw_scene()

    def _laplace_smoothing(self):

        if self.geometry is not None:

            l=0.5
            vertices_smooth = U.laplacian_smoothing(self.triangles, self.vertices, l)

            #setting the vertices to be the smoothed ones
            self.geometry.vertices = o3d.utility.Vector3dVector(vertices_smooth)

            #redrawing to see the difference
            self._redraw_scene()
            
    def _calc_eigenvectors(self):

        if self.geometry is not None:
            
            #calculating the graph laplacian
            L = U.graph_laplacian(self.triangles).astype(np.float32).toarray()
            
            #performing eigendecomposition
            vals, vecs = eigh(L)

            #sorting according to eigenvalue
            sort_idx = np.argsort(vals)
            self.eigenvectors = vecs[:, sort_idx]

    def _show_eigenvector(self):
        
        if self.eigenvectors is not None:
            
            # colors = np.zeros_like(self.vertices)
            
            scalars = self.eigenvectors[self.current_eigenvector]
            scalars = (scalars - scalars.min()) / (scalars.max() - scalars.min())

            # colors[:,0] = scalars
            colors = U.sample_colormap(scalars)

            self.geometry.vertex_colors = o3d.utility.Vector3dVector(colors)
            self._redraw_scene()
            
            
            
            #  =============           ERGASIA               ==============
            
            
    # erotima a
    def _taubin_smoothing(self):

        if self.geometry is not None:

            l=0.6
            m=0.4
            vertices_smooth = U.taubin_smoothing(self.triangles, self.vertices, l, m)

            #setting the vertices to be the smoothed ones and giving them new normals and colors
            self.geometry.vertices = o3d.utility.Vector3dVector(vertices_smooth)
            self.line_geometry = o3d.geometry.LineSet.create_from_triangle_mesh(self.geometry)
            self.geometry.compute_vertex_normals()
            vertex_normals = np.asarray(self.geometry.vertex_normals)
            vertex_colors = (vertex_normals+0.5) / 3
            self.geometry.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            
            #redrawing to see the difference
            self._redraw_scene()

    # erotima b
    def _taubin_smoothing_local(self):

        if self.geometry is not None:

            l=0.6
            m=0.4

            if self.selected_vertex is not None:
                neighbors = U.k_ring_adjacency(self.selected_vertex, self.triangles, 6)
           
            
            vertices_smooth = U.taubin_smoothing_local(self.triangles, self.vertices, neighbors, l, m)

            #setting the vertices to be the smoothed ones and giving them new normals and colors
            self.geometry.vertices = o3d.utility.Vector3dVector(vertices_smooth)
            self.line_geometry = o3d.geometry.LineSet.create_from_triangle_mesh(self.geometry)
            self.geometry.compute_vertex_normals()
            vertex_normals = np.asarray(self.geometry.vertex_normals)
            vertex_colors = (vertex_normals+0.5) / 3
            self.geometry.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            
            #redrawing to see the difference
            self._redraw_scene()

    # erotima c
    def _skeletonization(self):

        if self.geometry is not None:
            
            # average_face_area = self.geometry.get_surface_area()
                
            start = t.time()
            self.vertices, self.WL, self.WH, self.iterator, self.vertex_areas = U.skeletonization(
                self.triangles, self.vertices, self.WL, self.WH, self.iterator, self.vertex_areas)
            end = t.time()

            print("\ntime:", end-start)
            
            #setting the vertices to be the new ones and giving them new normals and colors
            self.geometry.vertices = o3d.utility.Vector3dVector(self.vertices)
            self.line_geometry = o3d.geometry.LineSet.create_from_triangle_mesh(self.geometry)
            self.geometry.compute_vertex_normals()
            vertex_normals = np.asarray(self.geometry.vertex_normals)
            vertex_colors = (vertex_normals+0.5) / 3
            self.geometry.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            #redrawing to see the difference
            self._redraw_scene()
    
    def _skeletonization_local(self):
        
        if self.geometry is not None:
            
            # average_face_area = self.geometry.get_surface_area()
            
            if self.selected_vertex is not None:
                neighbors = U.k_ring_adjacency(self.selected_vertex, self.triangles, 8)
                
            start = t.time()
            self.vertices, self.WL, self.WH, self.iterator, self.vertex_areas = U.skeletonization(
                self.triangles, self.vertices, self.WL, self.WH, self.iterator, self.vertex_areas, neighbors)
            end = t.time()

            print("\ntime:", end-start)
            
            #setting the vertices to be the new ones and giving them new normals and colors
            self.geometry.vertices = o3d.utility.Vector3dVector(self.vertices)
            self.line_geometry = o3d.geometry.LineSet.create_from_triangle_mesh(self.geometry)
            self.geometry.compute_vertex_normals()
            vertex_normals = np.asarray(self.geometry.vertex_normals)
            vertex_colors = (vertex_normals+0.5) / 3
            self.geometry.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

            #redrawing to see the difference
            self._redraw_scene()
            
    # erotima d
    def _pc_skeletonization(self):

        if self.geometry is not None:
            
            # Create a point cloud object from the vertices
            if (len(self.vertices)<100): 
                pcd = self.geometry.sample_points_poisson_disk(number_of_points=int(20*len(self.vertices)))
                radii = [0.1, 0.2, 0.4, 0.8, 0.16]
                
            elif (len(self.vertices)<1000): pcd = self.geometry.sample_points_poisson_disk(number_of_points=int(4*len(self.vertices)))

            elif (len(self.vertices)<2000): pcd = self.geometry.sample_points_poisson_disk(number_of_points=int(2*len(self.vertices)))

            else : pcd = self.geometry.sample_points_poisson_disk(number_of_points=int(1.5*len(self.vertices)))
            
            # Poisson surface reconstruction
            # mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth = 4)
            
            # Ball pivot surface reconstruction
            radii = [0.005, 0.01, 0.02, 0.04, 0.08]
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
            
            # Alpha shapes surface reconstruction
            # alpha = 0.3
            # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            
            # Access the array of triangles
            self.triangles = np.asarray(mesh.triangles)
            self.vertices = np.asarray(mesh.vertices)
            
            #setting the vertices and triangles to be the new ones and giving them new normals and colors
            self.geometry.vertices = o3d.utility.Vector3dVector(self.vertices)
            self.geometry.triangles = o3d.utility.Vector3iVector(self.triangles)
            self.line_geometry = o3d.geometry.LineSet.create_from_triangle_mesh(self.geometry)
            
            self.geometry.compute_vertex_normals()
            vertex_normals = np.asarray(self.geometry.vertex_normals)
            vertex_colors = (vertex_normals+0.5) / 3
            self.geometry.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            
            #redrawing to see the difference
            self._redraw_scene()

    # erotima e
    def _edge_decrease(self):

        if self.geometry is not None:
            
            print("\ntriangles before: ", len(self.triangles))
            start = t.time()
            
            if len(self.triangles)>2000:
                for i in range(10):
                    self.vertices, self.triangles = U.edge_collapse(self.triangles, self.vertices, int(len(self.triangles)/100)+1)
                    
            elif len(self.triangles)>500 and len(self.triangles)<=2000:
                for i in range(30):
                    self.vertices, self.triangles = U.edge_collapse(self.triangles, self.vertices, int(len(self.triangles)/100)+1)
                    
            else:
                self.vertices, self.triangles = U.edge_collapse(self.triangles, self.vertices, int(len(self.triangles)/10)+1)
            
            #setting the vertices and triangles to be the new ones and giving them new normals and colors 
            self.geometry.vertices = o3d.utility.Vector3dVector(self.vertices)
            self.geometry.triangles = o3d.utility.Vector3iVector(self.triangles)
            self.line_geometry = o3d.geometry.LineSet.create_from_triangle_mesh(self.geometry)
            self.geometry.compute_vertex_normals()
            vertex_normals = np.asarray(self.geometry.vertex_normals)
            vertex_colors = (vertex_normals+0.5) / 3
            self.geometry.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            
            end = t.time()
            print("triangles after: ",len(self.triangles), "time :", end-start)
            
            #redrawing to see the difference
            self._redraw_scene()
    
    def _edge_increase(self):

        if self.geometry is not None:
            
            print("\ntriangles before: ", len(self.triangles))
            start = t.time()
            
            self.vertices, self.triangles = U.edge_create(self.triangles, self.vertices)
            
            #setting the vertices and triangles to be the new ones and giving them new normals and colors
            self.geometry.vertices = o3d.utility.Vector3dVector(self.vertices)
            self.geometry.triangles = o3d.utility.Vector3iVector(self.triangles)
            self.line_geometry = o3d.geometry.LineSet.create_from_triangle_mesh(self.geometry)
            self.geometry.compute_vertex_normals()
            vertex_normals = np.asarray(self.geometry.vertex_normals)
            vertex_colors = (vertex_normals+0.5) / 3
            self.geometry.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            
            end = t.time()
            print("triangles after: ",len(self.triangles), "time :", end-start)
            
            #redrawing to see the difference
            self._redraw_scene()
    
    
    # erotima f
    def _coating_transfer(self):
        
        if self.geometry is not None:
            
            if self.selected_vertex is not None:
                if self.turn == 0 : neighbors = U.k_ring_adjacency(self.selected_vertex, self.triangles, 6)
                else : neighbors = U.k_ring_adjacency(self.selected_vertex, self.triangles, 6)
            
            if (self.turn == 0):
                
                self.coat, self.first_vector = U.extract_coating(self.triangles, self.vertices, neighbors, self.selected_vertex)
                self.turn = 1
                print("\ncoating extracted\n")

            else:

                self.vertices = U.apply_coating(self.triangles, self.vertices, neighbors, self.coat, self.first_vector, self.selected_vertex)
                self.turn = 0
                print("\ncoating applied\n")
                
                #setting the vertices and triangles to be the new ones and giving them new normals and colors
                self.geometry.vertices = o3d.utility.Vector3dVector(self.vertices)
                self.geometry.triangles = o3d.utility.Vector3iVector(self.triangles)
                self.line_geometry = o3d.geometry.LineSet.create_from_triangle_mesh(self.geometry)
                self.geometry.compute_vertex_normals()
                vertex_normals = np.asarray(self.geometry.vertex_normals)
                vertex_colors = (vertex_normals+0.5) / 3
                self.geometry.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            
                #redrawing to see the difference
                self._redraw_scene()
    
            #  =============        TELOS         ERGASIAS               ==============
            

def main():

    gui.Application.instance.initialize()

    w = AppWindow(1024, 768)
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            w.load(path)
        else:
            w.window.show_message_box("Error",
                                      "Could not open file '" + path + "'")

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()

if __name__ == "__main__":

    main()