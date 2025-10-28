namespace BookRent.User.Api;

internal static class UserFavouritesEndpoints
{
    internal static IResult GetFavourites(HttpRequest request)
    {
        
        return TypedResults.Ok("success");
    }
    
    internal static IResult AddFavourite(HttpRequest request)
    {
        
        return TypedResults.Ok("success");
    }
    
    internal static IResult RemoveFavourite(HttpRequest request)
    {
        
        return TypedResults.Ok("success");
    }
}