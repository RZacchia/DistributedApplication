namespace BookRent.User.Api;

internal static class UserEndpointsModule
{
    internal static void MapUserEndpoints(this IEndpointRouteBuilder app)
    {
        RouteGroupBuilder group = app.MapGroup("/user");

        group.MapGet("/userDetails", UserDataEndpoints.GetUser);
        group.MapGet("/favourites", UserFavouritesEndpoints.GetFavourites);
        group.MapPost("/addFavourite", UserFavouritesEndpoints.AddFavourite);
        group.MapDelete("/removeFavourite", UserFavouritesEndpoints.RemoveFavourite);
        
    }
}