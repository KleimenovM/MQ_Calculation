import astropy.table
import matplotlib.pyplot as plt
import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
from astroquery.simbad import Simbad
from matplotlib.colors import Normalize

from config.plotting import set_plotting_defaults, royalblue_palette, orangered_palette, save_figure

set_plotting_defaults()


def new_centers(center):
    shift = 5 * u.deg
    center1 = SkyCoord(l=center.l + shift, b=center.b + shift, frame='galactic')
    center2 = SkyCoord(l=center.l - shift, b=center.b + shift, frame='galactic')
    center3 = SkyCoord(l=center.l + shift, b=center.b - shift, frame='galactic')
    center4 = SkyCoord(l=center.l - shift, b=center.b - shift, frame='galactic')
    center5 = SkyCoord(l=center.l - 2 * shift, b=center.b + shift, frame='galactic')
    center6 = SkyCoord(l=center.l - 2 * shift, b=center.b - shift, frame='galactic')
    return [center1, center2, center3, center4, center5, center6]


def get_stars(center, centers):
    # Set center and size
    width = height = 10 * u.deg

    print("Querying stars...")
    stars_all = []
    for c in centers:
        # Query stars from Tycho-2 brighter than mag 6
        vizier = Vizier(columns=['_RAJ2000', '_DEJ2000', 'Vmag'], column_filters={"Vmag": "<6"})
        stars_all.append(vizier.query_region(c.icrs, width=width, height=height, catalog="V/50")[0])

    stars = astropy.table.vstack(stars_all)

    stars_gal = SkyCoord(stars["_RAJ2000"], stars["_DEJ2000"], frame='icrs').galactic
    lon, lat = stars_gal.l.deg, stars_gal.b.deg
    lon -= 360 * (lon > 180)
    return lon, lat, stars


def sagittarius_constellation():
    # sagittarius constellation
    Vizier.ROW_LIMIT = -1
    catalog = Vizier.get_catalogs('VI/49')[0]
    sgr = catalog[catalog['cst'] == 'SGR']
    coords = SkyCoord(ra=sgr['_RA.icrs'], dec=sgr['_DE.icrs'], frame='icrs')
    gal = coords.galactic
    sgr_l, sgr_b = gal.l.deg, gal.b.deg
    sgr_l -= 360 * (sgr_l > 180)

    teapot_star_names = {
        r"$\epsilon$": "Eps Sgr",
        r"$\delta$": "Del Sgr",
        r"$\lambda$": "Lam Sgr",
        r"$\phi$": "Phi Sgr",
        r"$\sigma$": "Sig Sgr",
        r"$\zeta$": "Zet Sgr",
        r"$\gamma^2$": "Gam2 Sgr",
        r"$\tau$": "Tau Sgr",
    }

    # Запрос координат из SIMBAD
    teapot_stars = {}
    for label, simbad_id in teapot_star_names.items():
        result = Simbad.query_object(simbad_id)
        ra = result["ra"][0]
        dec = result["dec"][0]
        coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
        teapot_stars[label] = coord

    connection_order = [
        r"$\epsilon$",
        r"$\delta$",
        r"$\lambda$",
        r"$\phi$",
        r"$\sigma$",
        r"$\tau$",
        r"$\zeta$",
        r"$\epsilon$",

        r"$\gamma^2$",
        r"$\delta$",

        r"$\phi$",
        r"$\zeta$",
    ]

    # Convert to galactic coordinates
    coords_gal = {name: coord.galactic for name, coord in teapot_stars.items()}
    l_vals = np.array([coords_gal[name].l.deg for name in connection_order])
    l_vals -= 360 * (l_vals > 180)
    b_vals = np.array([coords_gal[name].b.deg for name in connection_order])
    return sgr_l, sgr_b, l_vals, b_vals, coords_gal


def scorpius_constellation():
    # scorpiu constellation
    Vizier.ROW_LIMIT = -1
    catalog = Vizier.get_catalogs('VI/49')[0]
    sgr = catalog[catalog['cst'] == 'SCO']
    coords = SkyCoord(ra=sgr['_RA.icrs'], dec=sgr['_DE.icrs'], frame='icrs')
    gal = coords.galactic
    sco_l, sco_b = gal.l.deg, gal.b.deg
    sco_l -= 360 * (sco_l > 180)
    return sco_l, sco_b


def ngc_objects(center, centers):
    width = height = 10 * u.deg
    # NGC
    print("Querying NGC...")
    vizier = Vizier(columns=['_RAJ2000', '_DEJ2000', 'size'])
    ngc_all = []
    for c in centers:
        # Query stars from Tycho-2 brighter than mag 6
        vizier = Vizier(columns=['_RAJ2000', '_DEJ2000', 'size'], column_filters={"size": ">5"})
        ngc_all.append(vizier.query_region(c.icrs, width=width, height=height, catalog="VII/118")[0])
    ngc = astropy.table.vstack(ngc_all)
    ngc_gal = SkyCoord(ngc["_RAJ2000"], ngc["_DEJ2000"], frame='icrs').transform_to('galactic')
    lon_ngc, lat_ngc = ngc_gal.l.deg, ngc_gal.b.deg
    lon_ngc -= 360 * (lon_ngc > 180)
    return ngc, lon_ngc, lat_ngc


def mw_grad(ax):
    # Grid in galactic coordinates
    l = np.linspace(-180, 180, 1000)  # Longitude
    b = np.linspace(-90, 90, 500)  # Latitude
    L, B = np.meshgrid(l, b)

    # Fake "Milky Way" gradient: bright along b=0° (Galactic plane), dim away
    gradient = np.exp(-(B / 10) ** 2)  # Gaussian centered on b = 0, width ~10 deg

    ax.imshow(gradient, extent=[-180, 180, -90, 90],
              cmap='Purples', interpolation='bilinear',
              norm=Normalize(vmin=0, vmax=1), alpha=.2)
    return


def main():
    center = SkyCoord(l=6.8 * u.deg, b=-4.8 * u.deg, frame='galactic')
    centers = new_centers(center)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_facecolor("white")

    # all the stars
    lon, lat, stars = get_stars(center, centers)
    ax.scatter(lon, lat, s=(6 - stars["Vmag"]) ** 2,  # size inversely to mag
               marker="o", color="black", label=r'Stars $(m_V < 6)$')

    # scorpius constellation
    sco_l, sco_b = scorpius_constellation()
    ax.plot(sco_l, sco_b, linestyle='dotted', color='slategray', linewidth=1, alpha=1)
    ax.text(-3, -3, 'Scorpius', color='slategray')

    # sagittarius constellation
    sgr_l, sgr_b, l_vals, b_vals, coords_gal = sagittarius_constellation()

    ax.plot(sgr_l, sgr_b, linestyle='solid', color='slategray', linewidth=1, alpha=1)
    ax.text(7, -11, 'Sagittarius', color='slategray')
    ax.plot(l_vals, b_vals, color='seagreen', alpha=.4)
    for name, coord in coords_gal.items():
        if coord.b.deg > -16:
            l_mod = coord.l.deg - 360 * (coord.l.deg > 180)
            plt.text(l_mod, .5 + coord.b.deg, name, color='seagreen')

    # V4641
    ax.scatter(center.l.deg, center.b.deg, marker="+", color=orangered_palette[2])
    ax.text(1.2 * center.l.deg, 0.9 * center.b.deg, 'V4641 Sgr', color=orangered_palette[2])

    # NGC objects
    ngc, lon_ngc, lat_ngc = ngc_objects(center, centers)
    ax.scatter(lon_ngc, lat_ngc, s=ngc["size"], label='NGC objects',
               marker="o", color='royalblue', alpha=.5, edgecolors='none')

    # galactic plane & galactic center
    ax.plot([-20, 55], [0, 0], color=royalblue_palette[2])
    ax.text(12, .5, 'Galactic plane', color=royalblue_palette[2])
    ax.scatter(0, 0, marker="o", color=royalblue_palette[2])
    ax.text(3, .5, 'Galactic center', color=royalblue_palette[2])
    mw_grad(ax)

    ax.legend(loc='lower right')
    ax.set_xticks([-8, -4, 0, 4, 8, 12, 16], [352, 356, 0, 4, 8, 12, 16])
    ax.set_yticks([-16, -12, -8, -4, 0, 4])

    ax.set_xlabel("l, deg")
    ax.set_ylabel("b, deg")
    ax.set_aspect("equal")
    ax.invert_xaxis()
    ax.set_xlim(16, -8)
    ax.set_ylim(-16, 4)

    plt.tight_layout()
    save_figure("star_map")
    plt.show()
    return


if __name__ == '__main__':
    main()

