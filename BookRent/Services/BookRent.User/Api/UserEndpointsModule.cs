namespace BookRent.User.Api;

internal static class UserEndpointsModule
{
    internal static void MapUserEndpoints(this IEndpointRouteBuilder app)
    {
        RouteGroupBuilder group = app.MapGroup("/user");

        group.MapGet("/userDetails/{id:guid}", UserDataEndpoints.GetUser);
        group.MapPost("/userDetails/add", UserDataEndpoints.AddUserDetails);
        group.MapGet("/favourites/{id:guid}", UserFavouritesEndpoints.GetFavourites);
        group.MapPost("/addFavourite/", UserFavouritesEndpoints.AddFavourite);
        group.MapDelete("/removeFavourite/{userId:guid}/{bookId:guid}", UserFavouritesEndpoints.RemoveFavourite);
        group.MapDelete("/removeBooks/{bookId:guid}", UserFavouritesEndpoints.RemoveBook);

        
    }
}